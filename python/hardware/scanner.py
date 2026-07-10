"""
Scanner Routine
---------------
Manages both the coarse tissue prescan (Phase 1) and the high-resolution
tile scan (Phase 1+). The scan builds directly on the tissue map produced
by prescan.

Autofocus (Phase 2) hooks are stubbed in run_scan() — search for the
'PHASE 2 AF HOOK' comment to see exactly where they slot in.

Outputs written to /tmp/wsi_scan/:
    tile_r{row:03d}_c{col:03d}.jpg   — JPEG q80 tile from Pi camera
    scan_log.jsonl                   — Per-tile metadata (append-mode, crash-safe)
    scan_config.json                 — Scan parameters for downstream tools
    TileConfiguration.txt            — Fiji Grid/Collection Stitching import file
                                       (nominal pixel positions; calibrate PIXELS_PER_MM
                                        from your FOV measurement before using in Fiji)
"""

import asyncio
import time
import cv2
import numpy as np
import os
import json
import re
import shutil

from hardware.corrections import FlatFieldCorrector
from hardware.stitcher import Stitcher
# Nominal pixel scale used by the live stitch preview.
NOMINAL_PIXELS_PER_MM: float = 6627.0
CROP_MARGIN_PX: int = 500
AUTOFOCUS_MIN_TISSUE_FRACTION: float = 0.10
EDGE_TISSUE_FRACTION: float = 0.25  # below this, treat tile as tissue-edge — relax global gate

# Fiji tile coordinates are derived from tile row/col indices using the
# empirically measured affine offsets below.
FIJI_X_FROM_ROW: int = 380
FIJI_X_FROM_COL: int = -2194
FIJI_Y_FROM_COL: int = -304
FIJI_Y_FROM_ROW: int = -1494
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCAN_EXPORT_ROOT = os.path.join(PROJECT_ROOT, "scan_exports")


class ScannerRoutine:
    # AF trigger thresholds (read from AutoFocus config when available)
    _LOCAL_DROP_DEFAULT  = 0.05   # 5 % drop vs previous tile → run AF
    _GLOBAL_DROP_DEFAULT = 0.60   # 60 % drop vs global median → manual control
    _MIN_TISSUE_DEFAULT  = 0.03   # below this fraction → skip global gate

    def __init__(self, cnc, camera, led=None, canon=None, autofocus=None):
        """
        Args:
            cnc       : CNCController instance
            camera    : CameraClient instance
            led       : LedController instance  (optional; Phase 2 AF)
            canon     : CanonLens instance       (optional; Phase 2 AF)
            autofocus : AutoFocus instance       (optional; Phase 2 AF)
        """
        self.cnc       = cnc
        self.camera    = camera
        self.led       = led
        self.canon     = canon
        self.autofocus = autofocus

        self.is_scanning        = False
        self.event_queue        = asyncio.Queue()
        self.tissue_coordinates = []

        # ── Phase 2: per-scan AF state ─────────────────────────────────────────
        self.prev_tile_tenengrad:      float | None = None
        self.global_tenengrad_history: list[float]  = []
        self.global_tenengrad_median:  float        = 0.0

        # ── Pause / resume for manual focus intervention ───────────────────────
        self.is_paused    = False
        self.resume_event = asyncio.Event()

        self.is_paused = False
        self.resume_event = asyncio.Event()

        # ── Manual reimage review (end-of-scan focus audit) ────────────────────
        self.awaiting_reimage_review:      bool              = False
        self.reimage_decision_event:       asyncio.Event     = asyncio.Event()
        self.reimage_candidate_filenames:  list[str]         = []
        self.reimage_selected_filenames:   list[str] | None  = None

        # ── Stitcher (created fresh each run_scan call) ────────────────────────
        self.stitcher: Stitcher | None = None

        #FFC
        self.ffc: FlatFieldCorrector | None = None

    # ==================================================================
    # SHARED FOCUS METRICS
    # ==================================================================

    @staticmethod
    def tenengrad(gray_img: np.ndarray) -> float:
        """
        Tenengrad focus metric: mean of squared Sobel gradients.
        Higher value = sharper image. Used for Phase 2 hill-climb AF
        and as a tile-quality metric logged per tile in run_scan().
        """
        gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    @staticmethod
    def crop_vignette_edges(img: np.ndarray, margin_px: int = CROP_MARGIN_PX) -> np.ndarray:
        """
        Crop a fixed border from all sides to suppress edge vignetting.
        If the requested margin would remove the full image, return the input unchanged.
        """
        h, w = img.shape[:2]
        if margin_px <= 0 or (margin_px * 2) >= h or (margin_px * 2) >= w:
            return img
        return img[margin_px:h - margin_px, margin_px:w - margin_px]

    def _store_tile_image(
        self,
        img: np.ndarray,
        tile_path: str,
        row: int,
        col: int,
    ) -> tuple[float, str]:
        """
        Crop, save, score, and thumbnail a high-resolution tile image.
        Returns:
            (tenengrad_score, thumb_filename)
        """
        if self.ffc is not None:
            img = self.ffc.apply(img)

        img = self.crop_vignette_edges(img)

        if not cv2.imwrite(
            tile_path,
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ):
            raise RuntimeError(f"Failed to write tile image: {tile_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tenengrad_score = round(self.tenengrad(gray), 2)

        thumb_filename = f"thumb_r{row:03d}_c{col:03d}.jpg"
        thumb_path = os.path.join("/tmp/wsi_scan", thumb_filename)
        thumb = cv2.resize(img, (160, 120))
        cv2.imwrite(
            thumb_path,
            thumb,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        return tenengrad_score, thumb_filename

    @staticmethod
    def _export_tile_copy(src_path: str, export_dir: str, export_index: int) -> str:
        export_name = f"tile{export_index:03d}.jpg"
        export_path = os.path.join(export_dir, export_name)
        shutil.copyfile(src_path, export_path)
        return export_name

    @staticmethod
    def _fiji_tile_position(row: int, col: int) -> tuple[int, int]:
        """
        Convert tile row/col indices into Fiji tile coordinates.
        """
        px_x = (row * FIJI_X_FROM_ROW) + (col * FIJI_X_FROM_COL)
        px_y = (col * FIJI_Y_FROM_COL) + (row * FIJI_Y_FROM_ROW)
        return px_x, px_y

    def _assess_saved_tile_focus(self, tile_path: str) -> dict | None:
        """
        Load an already-saved tile JPEG, run tissue detection, and compute focus score.
        Tenengrad is intentionally recomputed from the saved JPEG-q95 file (not reused
        from the in-scan stream-frame log) — this is the archive-quality score and is
        what the audit/reimage decision should act on.
        """
        img = cv2.imread(tile_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        has_tissue, tissue_mask = self.tissue_present(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tissue_fraction = float(np.mean(tissue_mask))
        return {
            "has_tissue": has_tissue,
            "tissue_fraction": tissue_fraction,
            "is_edge_tile": tissue_fraction < EDGE_TISSUE_FRACTION,
            "tanengrad": round(self.tenengrad(gray), 2),
        }

    @staticmethod
    def _local_focus_outliers(
            tissue_tiles: list[dict],
            radius: int = 2,
            k: float = 2.5,
            min_neighbors: int = 4,
            global_floor_ratio: float = 0.5,
    ) -> tuple[list[dict], float]:
        """
        Flag tiles whose archive-quality Tenengrad is a statistical outlier
        relative to its immediate spatial neighbours, OR catastrophically
        low relative to the whole-slide median.

        Rationale: Tenengrad scales with how much real edge/texture content
        a region has (dense cellular tissue vs sparse stroma/fat), so a
        single global threshold over- or under-flags depending on local
        tissue type. Comparing each tile against its own neighbourhood
        (which usually shares similar tissue character) isolates genuine
        focus loss from natural texture variation. A neighbourhood median/MAD
        (robust to one or two other bad neighbours) drives a z-score test;
        tiles with too little local context fall back to the global floor,
        which also catches broad/systemic drift that a purely local test
        could individually under-flag (each tile in a drifted band still
        looks "normal" next to its equally-drifted neighbours) — by anchoring
        to the slide-wide median instead.

        Returns (flagged_tiles, global_median) — each flagged tile carries
        "flag_reason" ("local" / "global_floor" / "local+global_floor") and
        "local_reference" (neighbourhood median, or None if too few neighbours).
        """
        global_median = float(np.median([t["tanengrad"] for t in tissue_tiles]))
        by_pos = {(t["row"], t["col"]): t for t in tissue_tiles}

        flagged = []
        for t in tissue_tiles:
            r, c = t["row"], t["col"]
            neighbor_scores = [
                by_pos[(r + dr, c + dc)]["tanengrad"]
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if (dr, dc) != (0, 0)
                and (r + dr, c + dc) in by_pos
                and not by_pos[(r + dr, c + dc)]["is_edge_tile"]
            ]

            local_flag, local_median = False, None
            if len(neighbor_scores) >= min_neighbors:
                local_median = float(np.median(neighbor_scores))
                local_mad = float(np.median(
                    np.abs(np.array(neighbor_scores) - local_median)
                )) or 1e-6
                z = (local_median - t["tanengrad"]) / (1.4826 * local_mad)
                local_flag = z > k

            global_flag = (
                global_median > 0 and
                t["tanengrad"] < global_median * global_floor_ratio
            )

            if local_flag or global_flag:
                reasons = []
                if local_flag:  reasons.append("local")
                if global_flag: reasons.append("global_floor")
                flagged.append({
                    **t,
                    "flag_reason": "+".join(reasons),
                    "local_reference": round(local_median, 2) if local_median is not None else None,
                })

        return flagged, global_median

    async def _run_end_of_scan_focus_audit(
            self,
            loop,
            tile_lookup: dict[str, dict],
            feedrate: int,
            settle_time: float,
            log_path: str,
            total_tiles: int,
            captured_count: int,
            scan_step_mm: float,
            grid_cols: int,
            grid_rows: int,
            export_dir: str | None = None,
            export_lookup: dict[str, str] | None = None,
    ) -> dict:
        """
        Inspect saved tiles on disk, revisit tissue-bearing low-focus tiles,
        refocus, and overwrite the bad JPEGs.
        """
        tile_dir = "/tmp/wsi_scan"
        tile_pattern = re.compile(r"^tile_r\d{3}_c\d{3}\.jpg$")
        assessments = []

        for filename in sorted(os.listdir(tile_dir)):
            if not tile_pattern.match(filename):
                continue
            if filename not in tile_lookup:
                continue

            tile_path = os.path.join(tile_dir, filename)
            assessment = self._assess_saved_tile_focus(tile_path)
            if assessment is None:
                continue

            assessment.update(tile_lookup[filename])
            assessment["filename"] = filename
            assessments.append(assessment)

        tissue_tiles = [a for a in assessments if a["has_tissue"]]
        if not tissue_tiles:
            print("[Scan] End-of-scan focus audit skipped: no tissue-positive saved tiles.")
            return {
                "audited_tiles": len(assessments),
                "tissue_tiles": 0,
                "global_tanengrad_median": 0.0,
                "bad_tiles": 0,
                "reimaged_tiles": 0,
            }

        flagged_tiles, global_median = self._local_focus_outliers(tissue_tiles)

        print(f"[Scan] End-of-scan focus audit: {len(tissue_tiles)} tissue tiles, "
              f"global_median_T={global_median:.1f}, "
              f"revisit={len(flagged_tiles)} "
              f"(local={sum('local' in t['flag_reason'] for t in flagged_tiles)}, "
              f"global_floor={sum('global_floor' in t['flag_reason'] for t in flagged_tiles)})")

        # ── Manual review: let the user pick which flagged tiles get reimaged ──
        bad_tiles = flagged_tiles
        if flagged_tiles and self.is_scanning:
            bad_tiles = await self._review_reimage_candidates(
                flagged_tiles, total_tiles, grid_cols, grid_rows
            )
        user_skipped_count = len(flagged_tiles) - len(bad_tiles)

        reimaged_tiles = 0
        for idx, tile in enumerate(bad_tiles):
            if not self.is_scanning:
                print("[Scan] Focus audit stopped by user.")
                break

            row = tile["row"]
            col = tile["col"]
            x_mm = tile["x_mm"]
            y_mm = tile["y_mm"]
            filename = tile["filename"]
            tile_path = os.path.join(tile_dir, filename)

            await self.event_queue.put({
                "event": "scan_tile_moving",
                "col": col,
                "row": row,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "tile_index": total_tiles + idx,
                "total_tiles": total_tiles + len(bad_tiles),
                "step_mm": scan_step_mm,
                "grid_cols": grid_cols,
                "grid_rows": grid_rows,
                "reimage_pass": True,
            })

            await loop.run_in_executor(None, self.cnc.goto_xy, x_mm, y_mm, feedrate)
            await loop.run_in_executor(None, self.cnc.wait_for_idle_blocking)
            await asyncio.sleep(settle_time)

            if self.autofocus and self.canon:
                await self.autofocus.run(loop)

            snap_bytes = None
            try:
                snap_bytes = await self.camera.snap(timeout=15.0)
            except Exception as e:
                print(f"[Scan] Reimage SNAP error at ({x_mm:.3f}, {y_mm:.3f}): {e}")

            if not snap_bytes:
                continue

            try:
                arr = np.frombuffer(snap_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Failed to decode reimage SNAP JPEG")

                tenengrad_score, thumb_filename = self._store_tile_image(
                    img, tile_path, row, col
                )
                reimaged_tiles += 1
                export_filename = None
                if export_dir and export_lookup:
                    export_filename = export_lookup.get(filename)
                    if export_filename:
                        export_path = os.path.join(export_dir, export_filename)
                        shutil.copyfile(tile_path, export_path)

                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "event": "scan_tile_reimage",
                        "col": col,
                        "row": row,
                        "x_mm": x_mm,
                        "y_mm": y_mm,
                        "filename": filename,
                        "export_filename": export_filename,
                        "tenengrad_before": tile["tanengrad"],
                        "tenengrad_after": tenengrad_score,
                        "flag_reason": tile["flag_reason"],
                        "local_reference": tile["local_reference"],
                        "global_tanengrad_median": round(global_median, 2),
                        "timestamp": round(time.time(), 3),
                    }) + '\n')

                print(f"[Scan] Reimaged tile r{row:03d} c{col:03d}  "
                      f"T {tile['tanengrad']:.1f} -> {tenengrad_score:.1f}")

                await self.event_queue.put({
                    "event": "scan_tile",
                    "col": col,
                    "row": row,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "filename": filename,
                    "export_filename": export_filename,
                    "thumb_name": thumb_filename,
                    "captured": True,
                    "tenengrad": tenengrad_score,
                    "tenengrad_stream": None,
                    "af_triggered": bool(self.autofocus and self.canon),
                    "global_median": round(self.global_tenengrad_median, 2),
                    "tile_index": total_tiles + idx,
                    "total_tiles": total_tiles + len(bad_tiles),
                    "captured_count": captured_count,
                    "step_mm": scan_step_mm,
                    "grid_cols": grid_cols,
                    "grid_rows": grid_rows,
                    "reimaged": True,
                })
            except Exception as e:
                print(f"[Scan] Failed to reimage tile r{row:03d} c{col:03d}: {e}")

        return {
            "audited_tiles": len(assessments),
            "tissue_tiles": len(tissue_tiles),
            "global_tanengrad_median": round(global_median, 2),
            "bad_tiles": len(flagged_tiles),
            "user_selected_for_reimage": len(bad_tiles),
            "user_skipped": user_skipped_count,
            "reimaged_tiles": reimaged_tiles,
        }

    async def _review_reimage_candidates(
            self,
            flagged_tiles: list[dict],
            total_tiles: int,
            grid_cols: int,
            grid_rows: int,
    ) -> list[dict]:
        """
        Pause the end-of-scan audit and let the user pick which flagged
        tiles actually get revisited, via a checkbox list in the UI.
        All candidates arrive pre-checked client-side; the user unchecks
        any they're happy to leave as-is.

        Resumes when POST /api/scan/reimage/confirm sets reimage_decision_event
        (mirrors the existing manual-focus pause/resume_event pattern).
        """
        candidates = [
            {
                "filename":        t["filename"],
                "thumb_name":      f"thumb_r{t['row']:03d}_c{t['col']:03d}.jpg",
                "row":             t["row"],
                "col":             t["col"],
                "x_mm":            t["x_mm"],
                "y_mm":            t["y_mm"],
                "tanengrad":       t["tanengrad"],
                "tissue_fraction": round(t["tissue_fraction"], 4),
                "flag_reason":     t["flag_reason"],
                "local_reference": t["local_reference"],
            }
            for t in flagged_tiles
        ]

        self.reimage_candidate_filenames = [c["filename"] for c in candidates]
        self.reimage_selected_filenames  = None
        self.reimage_decision_event.clear()
        self.awaiting_reimage_review = True

        print(f"[Scan] Awaiting manual reimage selection for {len(candidates)} tile(s)...")

        await self.event_queue.put({
            "event":       "reimage_review_required",
            "candidates":  candidates,
            "count":       len(candidates),
            "total_tiles": total_tiles,
            "grid_cols":   grid_cols,
            "grid_rows":   grid_rows,
            "message":     f"{len(candidates)} tile(s) flagged for low focus — "
                           f"review and select which to reimage.",
        })

        await self.reimage_decision_event.wait()
        self.reimage_decision_event.clear()
        self.awaiting_reimage_review = False

        selected = set(self.reimage_selected_filenames or [])
        kept     = [t for t in flagged_tiles if t["filename"] in selected]
        skipped  = [t for t in flagged_tiles if t["filename"] not in selected]

        print(f"[Scan] Reimage selection received: {len(kept)} selected, "
              f"{len(skipped)} skipped by user.")

        await self.event_queue.put({
            "event":             "reimage_review_resolved",
            "selected_count":    len(kept),
            "skipped_count":     len(skipped),
            "skipped_filenames": [t["filename"] for t in skipped],
        })

        return kept

    # ==================================================================
    # PRESCAN  (unchanged from original)
    # ==================================================================

    def tissue_present(
        self, img,
        min_tissue_fraction:    float = 0.015,
        strong_tissue_fraction: float = 0.035,
        min_texture:            float = 6,
    ):
        """
        HSV colour filter + conditional Laplacian-variance texture check.

        Strategy (handles slight defocus during prescan):
          - tissue_fraction > strong_tissue_fraction (0.035): colour alone is enough —
            H&E pink/purple is clearly present even in a soft image; skip texture.
          - tissue_fraction > min_tissue_fraction (0.015): borderline — require texture
            confirmation (min_texture=6) to reject blank/debris tiles.
          - Below min_tissue_fraction: not tissue.

        Sensitivity tuning:
            min_tissue_fraction  0.02  → 0.015  (catch thinner/fainter tissue)
            strong_tissue_fraction 0.05 → 0.035 (trust colour sooner)
            min_texture          8     → 6      (allow softer tissue)
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1];  v = hsv[:, :, 2];  H = hsv[:, :, 0]

        pink_red_mask    = ((H >= 0) & (H <= 26)) | ((H >= 154) & (H <= 179))
        purple_blue_mask = (H >= 92) & (H <= 168)
        color_mask  = pink_red_mask | purple_blue_mask
        tissue_mask = color_mask & (s > 35) & (v < 248)

        tissue_fraction = float(np.mean(tissue_mask))

        # Strong colour signal — tissue confirmed even if slightly defocused
        if tissue_fraction > strong_tissue_fraction:
            return True, tissue_mask

        # Borderline colour — require texture to distinguish tissue from debris/ink
        if tissue_fraction > min_tissue_fraction:
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if texture > min_texture:
                return True, tissue_mask

        return False, tissue_mask

    async def run_prescan(self, step_mm=0.4, max_cols=20, max_rows=20):
        """Systematic snake-pattern grid prescan — identifies tissue tile positions."""
        self.is_scanning = True
        self.tissue_coordinates = []

        self.cnc.send_gcode("G91")
        await asyncio.sleep(0.5)

        print(f"\n[Prescan] Starting {max_cols}×{max_rows} grid (step={step_mm}mm)...")

        # Flush stale camera frames
        self.camera.new_frame_event.clear()
        self.camera.latest_frame = b''
        try:
            await asyncio.wait_for(self.camera.new_frame_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            print("[Prescan] Warning: frame sync timed out.")

        final_row_reached = 0

        for row in range(max_rows):
            if not self.is_scanning:
                break
            final_row_reached = row
            direction = 1 if (row % 2 == 0) else -1

            for col_idx in range(max_cols):
                if not self.is_scanning:
                    break

                current_col = col_idx if direction == 1 else (max_cols - 1 - col_idx)
                self.camera.new_frame_event.clear()
                self.camera.latest_frame = b''

                if col_idx == 0 and row == 0:
                    self.cnc.send_gcode(f"G1 X{step_mm * direction:.3f} F600")
                    self.cnc.send_gcode(f"G1 X{-step_mm * direction:.3f} F600")

                frame_bytes = None
                try:
                    await asyncio.wait_for(self.camera.new_frame_event.wait(), timeout=1.5)
                    frame_bytes = self.camera.latest_frame
                except asyncio.TimeoutError:
                    print(f"[Prescan] Frame timeout — Col:{current_col} Row:{row}")
                    frame_bytes = self.camera.latest_frame

                has_tissue = False
                logical_x  = current_col * step_mm
                logical_y  = row * step_mm

                if frame_bytes:
                    try:
                        img_arr = np.frombuffer(frame_bytes, np.uint8)
                        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            has_tissue, tissue_mask = self.tissue_present(img)
                            if has_tissue:
                                self.tissue_coordinates.append({
                                    "col":  current_col,
                                    "row":  row,
                                    "x_mm": logical_x,
                                    "y_mm": logical_y
                                })

                            mask_uint8 = (tissue_mask * 255).astype(np.uint8)
                            thumb = cv2.resize(img, (160, 120))
                            thumb_filename = f"tile_{current_col}_{row}.jpg"
                            cv2.imwrite(
                                os.path.join("/tmp/wsi_prescan", thumb_filename),
                                thumb,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                            )
                    except Exception as e:
                        print(f"[Prescan] Frame error: {e}")

                await self.event_queue.put({
                    "event":      "prescan_tile",
                    "col":        current_col,
                    "row":        row,
                    "x":          logical_x,
                    "y":          logical_y,
                    "has_tissue": has_tissue,
                    "thumb_name": f"tile_{current_col}_{row}.jpg",
                    "step_mm":    step_mm,
                    "grid_cols":  max_cols,
                    "grid_rows":  max_rows,
                })

                if col_idx < max_cols - 1:
                    self.cnc.send_gcode(f"G0 X{step_mm * direction:.3f} F500")
                    await asyncio.sleep(0.5)

            if row < max_rows - 1 and self.is_scanning:
                self.cnc.send_gcode(f"G1 Y{step_mm:.3f} F600")
                await asyncio.sleep(0.5)

        # Return to start
        if final_row_reached % 2 == 0:
            return_x = -((max_cols - 1) * step_mm)
            self.cnc.send_gcode(f"G0 X{return_x:.3f} F800")
            await asyncio.sleep(0.4)
        return_y = -(final_row_reached * step_mm)
        if return_y != 0:
            self.cnc.send_gcode(f"G0 Y{return_y:.3f} F800")
            await asyncio.sleep(0.4)

        self.save_tissue_map()

        bounds = None
        if self.tissue_coordinates:
            all_x = [t['x_mm'] for t in self.tissue_coordinates]
            all_y = [t['y_mm'] for t in self.tissue_coordinates]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            w = max(step_mm, max_x - min_x)
            h = max(step_mm, max_y - min_y)
            bounds = {
                "min_x": min_x - w * 0.05, "max_x": max_x + w * 0.05,
                "min_y": min_y - h * 0.05, "max_y": max_y + h * 0.05,
            }

        await self.event_queue.put({
            "event":   "prescan_complete",
            "bounds":  bounds,
            "step_mm": step_mm,
        })
        self.is_scanning = False
        print(f"[Prescan] Done. Tissue at {len(self.tissue_coordinates)} tiles.")

    def save_tissue_map(self):
        path = "/tmp/wsi_prescan/tissue_map.json"
        try:
            with open(path, 'w') as f:
                json.dump(self.tissue_coordinates, f, indent=4)
            print(f"[Prescan] tissue_map.json saved ({len(self.tissue_coordinates)} tiles).")
        except Exception as e:
            print(f"[Prescan] Failed to save tissue_map: {e}")

    # ==================================================================
    # SCAN — tile grid generation
    # ==================================================================

    # AFTER
    def _generate_scan_grid(
            self,
            tissue_map: list[dict],
            scan_step: float = 0.40,
            scan_step_y: float = 0.20,  # ← NEW: Y step; defaults to scan_step
            prescan_step: float = 0.40,
            scan_whole_rows: bool = False,
    ) -> tuple[list[dict], int, int]:
        """
        Build the high-resolution scan tile list in snake order.

        A scan tile is included only if it lies within `prescan_step` distance
        of at least one tissue-positive prescan tile.  This confines the scan to
        areas where tissue was actually detected, skipping blank glass tiles.

        The bounding box is still used to set grid dimensions so that column/row
        indices are consistent with the canvas coordinate system in the browser.
        Tiles outside the proximity threshold are simply excluded from the list.

        Returns:
            tiles      : list of {"row", "col", "x_mm", "y_mm"} in snake order
            grid_cols  : number of columns in the bounding-box grid
            grid_rows  : number of rows in the bounding-box grid
        """
        if not tissue_map:
            return [], 0, 0

        tissue_pos   = np.array([[t['x_mm'], t['y_mm']] for t in tissue_map])
        threshold_sq = prescan_step ** 2          # distance² threshold for inclusion

        # Bounding box with half-step padding
        min_x = float(tissue_pos[:, 0].min()) - scan_step * 0.5
        max_x = float(tissue_pos[:, 0].max()) + scan_step * 0.5
        min_y = float(tissue_pos[:, 1].min()) - scan_step * 0.5
        max_y = float(tissue_pos[:, 1].max()) + scan_step * 0.5

        def _arange(lo, hi, step):
            vals, v = [], float(lo)
            while v <= float(hi) + 1e-6:
                vals.append(round(v, 4))
                v += step
            return vals

        _step_y = scan_step_y if scan_step_y is not None else scan_step
        x_coords = _arange(min_x, max_x, scan_step)
        y_coords = _arange(min_y, max_y, _step_y)

        grid_cols = len(x_coords)
        grid_rows = len(y_coords)

        tiles = []
        for row_idx, y in enumerate(y_coords):
            qualifying_cols = set()
            for phys_col_idx, x in enumerate(x_coords):
                diffs   = tissue_pos - np.array([x, y])
                dist_sq = (diffs ** 2).sum(axis=1)
                if dist_sq.min() <= threshold_sq:
                    qualifying_cols.add(phys_col_idx)

            if row_idx % 2 == 0:
                col_order = list(enumerate(x_coords))
            else:
                col_order = list(reversed(list(enumerate(x_coords))))

            if scan_whole_rows:
                if not qualifying_cols:
                    continue
                included_cols = set(range(len(x_coords)))
            else:
                included_cols = qualifying_cols

            for phys_col_idx, x in col_order:
                if phys_col_idx in included_cols:
                    tiles.append({
                        "row":   row_idx,
                        "col":   phys_col_idx,
                        "x_mm":  x,
                        "y_mm":  y,
                    })

        return tiles, grid_cols, grid_rows

    @staticmethod
    def _generate_full_prescan_map(
            prescan_step_mm: float,
            prescan_max_cols: int,
            prescan_max_rows: int,
    ) -> list[dict]:
        """
        Synthesize a full prescan tissue map by marking every prescan tile as tissue.
        This is used when the operator wants to skip prescan and scan the full area.
        """
        tiles = []
        for row in range(prescan_max_rows):
            for col in range(prescan_max_cols):
                tiles.append({
                    "col": col,
                    "row": row,
                    "x_mm": col * prescan_step_mm,
                    "y_mm": row * prescan_step_mm,
                })
        return tiles

    # ==================================================================
    # SCAN — main acquisition loop
    # ==================================================================

    # AFTER
    async def run_scan(
            self,
            scan_step_mm: float = 0.40,
            scan_step_y_mm: float = 0.25,  # ← NEW: smaller Y step for row overlap
            prescan_step_mm: float = 0.40,
            skip_prescan: bool = False,
            scan_whole_rows: bool = False,
            prescan_max_cols: int = 20,
            prescan_max_rows: int = 20,
            settle_time: float = 0.6,
            feedrate: int = 400,
    ):
        """
        High-resolution tile acquisition.

        Prerequisites:
            1. Either:
               - run_prescan() has completed and tissue_map.json exists, or
               - skip_prescan=True, in which case the full prescan area is scanned.
            2. User has manually moved the CNC back to the scan origin
               (same position used at the start of prescan) and confirmed
               via the 'Start High-Res Scan' button.  The endpoint calls
               cnc.set_origin() before launching this background task.

        Phase 2 autofocus will be injected at the labelled hook inside the
        per-tile loop.  Everything else in this method is Phase 1.
        """
        if self.is_scanning:
            return

        self.is_scanning  = True
        scan_start_time   = time.time()
        os.makedirs("/tmp/wsi_scan", exist_ok=True)

        # Reset per-scan AF tracking
        self.prev_tile_tenengrad       = None
        self.global_tenengrad_history  = []
        self.global_tenengrad_median   = 0.0
        self.is_paused                 = False
        self.resume_event.clear()

        # ── Load or synthesize tissue map ───────────────────────────────
        if skip_prescan:
            tissue_map = self._generate_full_prescan_map(
                prescan_step_mm=prescan_step_mm,
                prescan_max_cols=prescan_max_cols,
                prescan_max_rows=prescan_max_rows,
            )
            print(f"[Scan] Prescan skipped; scanning full {prescan_max_cols}×{prescan_max_rows} prescan area.")
        else:
            tissue_map_path = "/tmp/wsi_prescan/tissue_map.json"
            try:
                with open(tissue_map_path) as f:
                    tissue_map = json.load(f)
            except FileNotFoundError:
                await self.event_queue.put({
                    "event":   "scan_error",
                    "message": "tissue_map.json not found — run prescan first or enable skip_prescan.",
                })
                self.is_scanning = False
                return

            if not tissue_map:
                await self.event_queue.put({
                    "event":   "scan_error",
                    "message": "No tissue found in prescan. Check illumination, run prescan again, or enable skip_prescan.",
                })
                self.is_scanning = False
                return

        # ── Generate tile list ───────────────────────────────────────────
        tiles, grid_cols, grid_rows = self._generate_scan_grid(
            tissue_map, scan_step_mm, scan_step_y_mm, prescan_step_mm, scan_whole_rows
        )
        total_tiles = len(tiles)
        tile_lookup = {
            f"tile_r{tile['row']:03d}_c{tile['col']:03d}.jpg": tile
            for tile in tiles
        }

        if total_tiles == 0:
            await self.event_queue.put({
                "event":   "scan_error",
                "message": "Scan grid generated 0 tiles. Check prescan tissue map.",
            })
            self.is_scanning = False
            return

        print(f"\n[Scan] Starting high-res scan: {total_tiles} tiles  "
              f"grid={grid_cols}×{grid_rows}  step={scan_step_mm}mm")

        # Infer prescan grid dimensions from the tissue map for frontend coordinate mapping.
        # This lets the canvas work correctly even if the user reloaded the page since prescan.
        prescan_max_col = max(t['col'] for t in tissue_map)
        prescan_max_row = max(t['row'] for t in tissue_map)

        # Compact tile list for the frontend to pre-populate the canvas
        tile_manifest = [
            {"col": t["col"], "row": t["row"], "x_mm": t["x_mm"], "y_mm": t["y_mm"]}
            for t in tiles
        ]

        await self.event_queue.put({
            "event":            "scan_started",
            "total_tiles":      total_tiles,
            "grid_cols":        grid_cols,
            "grid_rows":        grid_rows,
            "step_mm":          scan_step_mm,
            "skip_prescan":     skip_prescan,
            "scan_whole_rows":  scan_whole_rows,
            "prescan_step_mm":  prescan_step_mm,
            "prescan_grid_cols": prescan_max_col + 1,
            "prescan_grid_rows": prescan_max_row + 1,
            "tiles":            tile_manifest,
        })

        # ── Activate on-axis illumination for capture ────────────────────
        # Phase 2 will toggle off-axis for AF frames, then back on-axis here.
        if self.led:
            await self.led.async_on_axis()

        # ── CNC: switch to absolute positioning mode ─────────────────────
        # set_origin() was already called by the API endpoint; this is a guard.
        loop = asyncio.get_running_loop()

        # ── Logging setup ────────────────────────────────────────────────
        log_path         = "/tmp/wsi_scan/scan_log.jsonl"
        tile_config_rows = []    # Fiji TileConfiguration lines
        captured_count   = 0
        export_dir       = None
        export_manifest  = []
        failed_snap_tiles: list[dict] = []
        export_count     = 0

        if scan_whole_rows:
            os.makedirs(SCAN_EXPORT_ROOT, exist_ok=True)
            export_dir = os.path.join(
                SCAN_EXPORT_ROOT,
                time.strftime("scan_%Y%m%d_%H%M%S"),
            )
            os.makedirs(export_dir, exist_ok=True)

        # ── Initialise incremental stitcher ──────────────────────────────────
        self.stitcher = Stitcher(
            pixels_per_mm=NOMINAL_PIXELS_PER_MM,
            step_mm=scan_step_mm,
            step_y_mm=scan_step_y_mm,  # ← pass independent Y step
        )

        # ── Per-tile acquisition loop ────────────────────────────────────
        for tile_idx, tile in enumerate(tiles):
            if not self.is_scanning:
                print("[Scan] Stopped by user.")
                break

            x_mm = tile["x_mm"]
            y_mm = tile["y_mm"]
            row  = tile["row"]
            col  = tile["col"]
            filename = f"tile_r{row:03d}_c{col:03d}.jpg"
            tile_path = os.path.join("/tmp/wsi_scan", filename)

            # Notify UI: currently moving to this tile
            await self.event_queue.put({
                "event":         "scan_tile_moving",
                "col":           col,
                "row":           row,
                "x_mm":          x_mm,
                "y_mm":          y_mm,
                "tile_index":    tile_idx,
                "total_tiles":   total_tiles,
                "step_mm":       scan_step_mm,
                "grid_cols":     grid_cols,
                "grid_rows":     grid_rows,
            })

            # Move CNC to tile (blocking in executor, does not freeze event loop)
            await loop.run_in_executor(None, self.cnc.goto_xy, x_mm, y_mm, feedrate)
            await loop.run_in_executor(None, self.cnc.wait_for_idle_blocking)
            await asyncio.sleep(settle_time)

            # ══════════════════════════════════════════════════════════════
            # PHASE 2 — AUTOFOCUS LOGIC
            # ══════════════════════════════════════════════════════════════
            af_triggered    = False
            T_before_af     = 0.0
            T_stream        = 0.0          # on-axis stream Tenengrad (used for gates)

            # Thresholds: prefer AutoFocus config values, fall back to class defaults
            af_cfg       = self.autofocus.config if self.autofocus else None
            local_drop   = af_cfg.local_drop  if af_cfg else self._LOCAL_DROP_DEFAULT
            global_drop  = af_cfg.global_drop if af_cfg else self._GLOBAL_DROP_DEFAULT
            global_win   = af_cfg.global_window if af_cfg else 30

            # ── 1. Grab on-axis stream frame for focus quality check ─────────
            self.camera.new_frame_event.clear()
            self.camera.latest_frame = b''
            try:
                await asyncio.wait_for(self.camera.new_frame_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

            check_frame = self.camera.latest_frame
            tissue_fraction = 0.0
            has_tissue      = False

            if check_frame:
                arr       = np.frombuffer(check_frame, np.uint8)
                check_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if check_bgr is not None:
                    gray_check = cv2.cvtColor(check_bgr, cv2.COLOR_BGR2GRAY)
                    T_stream   = self.tenengrad(gray_check)
                    T_before_af = T_stream

                    # Tissue fraction check (reuse prescan HSV logic)
                    _, tissue_mask  = self.tissue_present(check_bgr)
                    tissue_fraction = float(np.mean(tissue_mask))
                    has_tissue      = tissue_fraction > AUTOFOCUS_MIN_TISSUE_FRACTION
                    is_edge_tile = tissue_fraction < EDGE_TISSUE_FRACTION

            # ── 2. Evaluate global failure gate ──────────────────────────────
            global_gate = (has_tissue and not is_edge_tile and
                           self.global_tenengrad_median > 0 and
                           T_stream < self.global_tenengrad_median * (1.0 - global_drop))

            # ── 3. Run AF only on tiles with enough tissue coverage ──────────
            if self.autofocus and self.canon and has_tissue:
                af_triggered = True
                T_stream     = await self.autofocus.run(loop)

            # ── 4. Global gate re-check after AF ─────────────────────────────
            #    If Tenengrad is STILL catastrophically low after AF, try a
            #    coarse CNC Z recovery followed by another Canon ring correction
            #    before pausing for manual input.
            if (global_gate and has_tissue and self.global_tenengrad_median > 0 and
                    T_stream < self.global_tenengrad_median * (1.0 - global_drop) and
                    self.autofocus and self.cnc and self.canon):
                T_stream = await self.autofocus.recover_low_global_focus(loop)

            # ── 5. Manual pause only if recovery still failed ─────────────────
            if (global_gate and has_tissue and self.global_tenengrad_median > 0 and
                    T_stream < self.global_tenengrad_median * (1.0 - global_drop)):
                await self._pause_for_manual_focus(
                    tile_idx, col, row, x_mm, y_mm, T_stream)

            # ── 6. Update tracking state ─────────────────────────────────────
            if has_tissue and not is_edge_tile:                   # exclude blank/edge tiles from global avg
                self.global_tenengrad_history.append(T_stream)
                if len(self.global_tenengrad_history) > global_win:
                    self.global_tenengrad_history.pop(0)
                self.global_tenengrad_median = float(
                    np.median(self.global_tenengrad_history))

            self.prev_tile_tenengrad = T_stream
            # ══════════════════════════════════════════════════════════════
            # END PHASE 2 AUTOFOCUS
            # ══════════════════════════════════════════════════════════════

            # ── Post-AF mechanical settle ────────────────────────────────
            if af_triggered:
                await asyncio.sleep(0.21)

            # ── Capture high-res SNAP ────────────────────────────────────
            snap_bytes = None
            try:
                snap_bytes = await self.camera.snap(timeout=15.0)
            except Exception as e:
                print(f"[Scan] SNAP error at ({x_mm:.3f}, {y_mm:.3f}): {e}")

            # ── Save tile and log metadata ───────────────────────────────
            tenengrad_score = 0.0
            thumb_filename  = None
            tile_saved      = False
            export_filename = None
            # Reserve the export slot unconditionally — failed SNAPs hold their index
            reserved_export_index = None
            if export_dir:
                export_count += 1
                reserved_export_index = export_count
            if snap_bytes:
                try:
                    arr = np.frombuffer(snap_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        tenengrad_score, thumb_filename = self._store_tile_image(
                            img, tile_path, row, col
                        )
                        tile_saved = True
                        captured_count += 1
                        if export_dir:

                            export_filename = self._export_tile_copy(
                                tile_path, export_dir, reserved_export_index
                            )
                            export_manifest.append({
                                "tile_index": tile_idx,
                                "export_index": export_count,
                                "export_filename": export_filename,
                                "source_filename": filename,
                                "export_index": reserved_export_index,
                                "row": row,
                                "col": col,
                                "x_mm": x_mm,
                                "y_mm": y_mm,
                            })
                    else:
                        raise RuntimeError("Failed to decode SNAP JPEG")
                except Exception:
                    print(f"[Scan] Failed to process SNAP tile at ({x_mm:.3f}, {y_mm:.3f})")

                    # ── Incremental stitch ───────────────────────────────────────────
                    # Runs in a thread pool — never blocks the acquisition loop.
                    # Thumbnail must be written first (generate_preview reads it).
                # Thumbnail must be written first (generate_preview reads it).
                if self.stitcher:
                    # The camera image X-axis is inverted relative to CNC X
                    # (canvas draws pxX = width - x_mm*scale).  Flip col so the stitcher's
                    # internal coordinate frame matches the camera image orientation.
                    _stitch_col = (grid_cols - 1) - col
                    await self.stitcher.on_tile_captured(row, _stitch_col, x_mm, y_mm, filename)

            if tile_saved:
                # Append to JSONL log (crash-safe)
                # NOTE: tenengrad_snap and tenengrad_stream are NOT comparable.
                #   tenengrad_snap   — computed on the 4056×3040 SNAP JPEG (archive quality metric)
                #   tenengrad_stream — computed on 640×480 stream frames (drives AF decisions)
                # The Tenengrad scale differs by resolution; never compare them directly.
                tile_meta = {
                    "col":                    col,
                    "row":                    row,
                    "x_mm":                   x_mm,
                    "y_mm":                   y_mm,
                    "filename":               filename,
                    "export_filename":         export_filename,
                    "tenengrad_snap":         tenengrad_score,       # from SNAP (4056×3040)
                    "tenengrad_stream_pre":   round(T_before_af, 2), # stream T before AF
                    "tenengrad_stream_post":  round(T_stream, 2),    # stream T after AF (used for tracking)
                    "tissue_fraction": round(tissue_fraction, 4),
                    "is_edge_tile": is_edge_tile,
                    "af_triggered":           af_triggered,
                    "global_median":          round(self.global_tenengrad_median, 2),
                    "tile_index":             tile_idx,
                    "crop_margin_px":         CROP_MARGIN_PX,
                    "timestamp":              round(time.time(), 3),
                }
                with open(log_path, 'a') as f:
                    f.write(json.dumps(tile_meta) + '\n')

                px_x, px_y = self._fiji_tile_position(row, col)
                tile_config_rows.append(f"{filename}; ; ({px_x}, {px_y})")

            if not tile_saved:
                failed_snap_tiles.append({
                    "row": row, "col": col,
                    "x_mm": x_mm, "y_mm": y_mm,
                    "filename": filename,
                    "reserved_export_index": reserved_export_index,
                })

            status = "captured" if tile_saved else "failed"
            af_tag = " [AF]" if af_triggered else ""
            print(f"[Scan] [{tile_idx+1:4d}/{total_tiles}] "
                  f"({x_mm:.3f}, {y_mm:.3f})mm  "
                  f"stream_T={T_stream:.0f}  snap_T={tenengrad_score:.0f}  "
                  f"median={self.global_tenengrad_median:.0f}{af_tag}  {status}")

            # Notify UI: tile result (thumb_name lets browser load the canvas preview image)
            await self.event_queue.put({
                "event":                  "scan_tile",
                "col":                    col,
                "row":                    row,
                "x_mm":                   x_mm,
                "y_mm":                   y_mm,
                "filename":               filename,
                "export_filename":        export_filename,
                "thumb_name":             thumb_filename,
                "captured":               tile_saved,
                "tenengrad":              tenengrad_score,          # SNAP (4056×3040)
                "tenengrad_stream":       round(T_stream, 2),       # stream post-AF (drives global gate)
                "af_triggered":           af_triggered,
                "global_median":          round(self.global_tenengrad_median, 2),
                "tile_index":             tile_idx,
                "total_tiles":            total_tiles,
                "captured_count":         captured_count,
                "step_mm":                scan_step_mm,
                "grid_cols":              grid_cols,
                "grid_rows":              grid_rows,
            })

        # ── Re-snap failed tiles (mandatory — no user review) ───────────────
        if failed_snap_tiles and self.is_scanning:
            print(f"[Scan] Re-snapping {len(failed_snap_tiles)} failed tile(s)...")
            for tile in failed_snap_tiles:
                if not self.is_scanning:
                    break

                row, col = tile["row"], tile["col"]
                x_mm, y_mm = tile["x_mm"], tile["y_mm"]
                filename = tile["filename"]
                tile_path = os.path.join("/tmp/wsi_scan", filename)
                ridx = tile["reserved_export_index"]

                await self.event_queue.put({
                    "event": "scan_tile_moving",
                    "col": col, "row": row,
                    "x_mm": x_mm, "y_mm": y_mm,
                    "tile_index": total_tiles,
                    "total_tiles": total_tiles,
                    "step_mm": scan_step_mm,
                    "grid_cols": grid_cols, "grid_rows": grid_rows,
                    "reimage_pass": True,
                })

                await loop.run_in_executor(None, self.cnc.goto_xy, x_mm, y_mm, feedrate)
                await loop.run_in_executor(None, self.cnc.wait_for_idle_blocking)
                await asyncio.sleep(settle_time)

                if self.autofocus and self.canon:
                    await self.autofocus.run(loop)

                snap_bytes = None
                try:
                    snap_bytes = await self.camera.snap(timeout=15.0)
                except Exception as e:
                    print(f"[Scan] Re-snap error r{row:03d}c{col:03d}: {e}")

                if not snap_bytes:
                    print(f"[Scan] Re-snap permanently failed r{row:03d}c{col:03d} "
                          f"— export slot tile{ridx:03d}.jpg will be absent")
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            "event": "snap_failed_permanent",
                            "col": col, "row": row,
                            "filename": filename,
                            "reserved_export_index": ridx,
                            "timestamp": round(time.time(), 3),
                        }) + '\n')
                    continue

                try:
                    arr = np.frombuffer(snap_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("decode failed")

                    tenengrad_score, thumb_filename = self._store_tile_image(
                        img, tile_path, row, col
                    )
                    captured_count += 1

                    export_filename = None
                    if export_dir and ridx is not None:
                        export_filename = self._export_tile_copy(tile_path, export_dir, ridx)
                        export_manifest.append({
                            "tile_index": None,  # original tile_idx not tracked here
                            "export_index": ridx,
                            "export_filename": export_filename,
                            "source_filename": filename,
                            "row": row, "col": col,
                            "x_mm": x_mm, "y_mm": y_mm,
                            "snap_recovered": True,
                        })

                    px_x, px_y = self._fiji_tile_position(row, col)
                    tile_config_rows.append(f"{filename}; ; ({px_x}, {px_y})")

                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            "event": "snap_failed_recovered",
                            "col": col, "row": row,
                            "filename": filename,
                            "export_filename": export_filename,
                            "export_index": ridx,
                            "tenengrad_snap": tenengrad_score,
                            "timestamp": round(time.time(), 3),
                        }) + '\n')

                    await self.event_queue.put({
                        "event": "scan_tile",
                        "col": col, "row": row,
                        "x_mm": x_mm, "y_mm": y_mm,
                        "filename": filename,
                        "export_filename": export_filename,
                        "thumb_name": thumb_filename,
                        "captured": True,
                        "tenengrad": tenengrad_score,
                        "tenengrad_stream": None,
                        "af_triggered": bool(self.autofocus and self.canon),
                        "global_median": round(self.global_tenengrad_median, 2),
                        "tile_index": total_tiles,
                        "total_tiles": total_tiles,
                        "captured_count": captured_count,
                        "step_mm": scan_step_mm,
                        "grid_cols": grid_cols, "grid_rows": grid_rows,
                        "reimaged": True,
                    })

                except Exception as e:
                    print(f"[Scan] Re-snap processing failed r{row:03d}c{col:03d}: {e}")


        export_lookup = {
            e["source_filename"]: e["export_filename"]
            for e in export_manifest
        }
        remediation_summary = await self._run_end_of_scan_focus_audit(
            loop=loop,
            tile_lookup=tile_lookup,
            feedrate=feedrate,
            settle_time=settle_time,
            log_path=log_path,
            total_tiles=total_tiles,
            captured_count=captured_count,
            scan_step_mm=scan_step_mm,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            export_dir=export_dir,
            export_lookup=export_lookup,
        )

        # ── Return to origin ─────────────────────────────────────────────
        print("[Scan] Returning to origin...")
        await loop.run_in_executor(None, self.cnc.return_to_origin, 600)
        await loop.run_in_executor(None, self.cnc.wait_for_idle_blocking)
        await loop.run_in_executor(None, self.cnc.restore_relative_mode)

        # ── Write Fiji TileConfiguration.txt ────────────────────────────
        self._write_tile_config(tile_config_rows, scan_step_mm)

        # ── Write scan_config.json ───────────────────────────────────────
        scan_config = {
            "scan_step_mm":          scan_step_mm,
            "prescan_step_mm":       prescan_step_mm,
            "skip_prescan":          skip_prescan,
            "scan_whole_rows":       scan_whole_rows,
            "export_dir":            export_dir,
            "prescan_grid_cols":     prescan_max_col + 1,
            "prescan_grid_rows":     prescan_max_row + 1,
            "nominal_pixels_per_mm": NOMINAL_PIXELS_PER_MM,
            "crop_margin_px":        CROP_MARGIN_PX,
            "total_tiles":           total_tiles,
            "captured_tiles":        captured_count,
            "duration_seconds":      round(time.time() - scan_start_time, 1),
            "focus_remediation":     remediation_summary,
            "fiji_tile_transform": {
                "x": "row*380 + col*(-2194)",
                "y": "col*(-304) + row*(-1494)",
            },
        }
        with open("/tmp/wsi_scan/scan_config.json", 'w') as f:
            json.dump(scan_config, f, indent=4)

        if export_dir:
            with open(os.path.join(export_dir, "manifest.json"), 'w') as f:
                json.dump({
                    "scan_whole_rows": True,
                    "total_exports": export_count,
                    "tiles": export_manifest,
                }, f, indent=2)

        duration = round(time.time() - scan_start_time, 1)
        await self.event_queue.put({
            "event":            "scan_complete",
            "total_tiles":      total_tiles,
            "captured_tiles":   captured_count,
            "duration_seconds": duration,
            "focus_remediation": remediation_summary,
        })

        self.is_scanning = False
        print(f"\n[Scan] Complete: {captured_count}/{total_tiles} tiles  "
              f"in {duration:.0f}s  ({duration/max(captured_count,1):.1f}s/tile)")

    # ==================================================================
    # SCAN — Fiji TileConfiguration writer
    # ==================================================================

    def _write_tile_config(self, rows: list[str], scan_step_mm: float):
        """
        Write a Fiji Grid/Collection Stitching compatible TileConfiguration.txt.

        To use in Fiji:
            Plugins → Stitch → Grid/Collection Stitching
            Type: Positions from file
            Browse to: /tmp/wsi_scan/TileConfiguration.txt

        Pixel positions here use the fixed row/col transform configured for Fiji.
        """
        path = "/tmp/wsi_scan/TileConfiguration.txt"
        header = (
            "# Fiji Grid/Collection Stitching — Positions from file\n"
            f"# Scan step: {scan_step_mm} mm\n"
            f"# X = row*{FIJI_X_FROM_ROW} + col*({FIJI_X_FROM_COL})\n"
            f"# Y = col*({FIJI_Y_FROM_COL}) + row*({FIJI_Y_FROM_ROW})\n"
            "# dim = 2\n"
        )
        with open(path, 'w') as f:
            f.write(header)
            for line in rows:
                f.write(line + '\n')
        print(f"[Scan] TileConfiguration.txt written ({len(rows)} tiles).")

    # ==================================================================
    # SCAN — Tile config regenerator (call after measuring actual FOV)
    # ==================================================================

    @staticmethod
    def regenerate_tileconfig(log_path: str = "/tmp/wsi_scan/scan_log.jsonl"):
        """
        Re-generate TileConfiguration.txt from saved tile row/col metadata.
        """
        rows = []
        try:
            with open(log_path) as f:
                for line in f:
                    t = json.loads(line.strip())
                    px_x, px_y = ScannerRoutine._fiji_tile_position(t["row"], t["col"])
                    rows.append(f"{t['filename']}; ; ({px_x}, {px_y})")
        except FileNotFoundError:
            print(f"[Scan] scan_log.jsonl not found at {log_path}")
            return

        out_path = "/tmp/wsi_scan/TileConfiguration.txt"
        with open(out_path, 'w') as f:
            f.write("# Fiji Grid/Collection Stitching — Positions from file\n")
            f.write(f"# X = row*{FIJI_X_FROM_ROW} + col*({FIJI_X_FROM_COL})\n")
            f.write(f"# Y = col*({FIJI_Y_FROM_COL}) + row*({FIJI_Y_FROM_ROW})\n")
            f.write("dim = 2\n")
            for row in rows:
                f.write(row + '\n')
        print(f"[Scan] Regenerated TileConfiguration.txt ({len(rows)} tiles)")

    # ==================================================================
    # PHASE 2 — Manual focus pause helper
    # ==================================================================

    async def _pause_for_manual_focus(self, tile_idx, col, row, x_mm, y_mm, tenengrad):
        """
        Pause the scan and wait for the user to manually re-focus.
        Resumes when /api/scan/resume is called (sets self.resume_event).
        """
        self.is_paused = True
        self.resume_event.clear()
        print(f"\n[Scan] ⚠  Focus failure at tile {tile_idx} "
              f"({x_mm:.2f}, {y_mm:.2f})mm  "
              f"T={tenengrad:.0f}  global_median={self.global_tenengrad_median:.0f}")
        print("[Scan]    Paused — waiting for manual re-focus and /api/scan/resume")

        await self.event_queue.put({
            "event":         "scan_paused",
            "reason":        "focus_failure",
            "tile_index":    tile_idx,
            "col":           col,
            "row":           row,
            "x_mm":          x_mm,
            "y_mm":          y_mm,
            "tenengrad":     round(tenengrad, 1),
            "global_median": round(self.global_tenengrad_median, 1),
            "message":       "AF could not recover focus. Adjust manually, then click Resume.",
        })

        await self.resume_event.wait()
        self.resume_event.clear()
        self.is_paused = False
        print("[Scan] Resumed.")

    # ==================================================================
    # Stop (works for both prescan and scan)
    # ==================================================================

    def stop(self):
        self.is_scanning = False
        if self.awaiting_reimage_review:
            self.reimage_selected_filenames = []
            self.reimage_decision_event.set()
        print("[Scanner] Stop signal sent.")
