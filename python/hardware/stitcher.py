"""
hardware/stitcher.py — GPU-accelerated incremental WSI tile stitcher
---------------------------------------------------------------------
Phase correlation (cuFFT) on overlap strips for pairwise registration,
with warm-started Jacobi spring-mesh relaxation for global consistency.
Older tiles shift when new neighbours add contradicting constraints.

Scan geometry (calibrate NOMINAL_PIXELS_PER_MM in scanner.py first):
    step_mm   = 0.3 mm
    crop_px   = 500 px from each edge  (removes objective lens artefacts)
    eff_w     = 4056 − 2×500 = 3056 px  (effective tile width)
    eff_h     = 3040 − 2×500 = 2040 px  (effective tile height)

    At NOMINAL 10 000 px/mm:  step_px = 3000  →  ov_x = 3056−3000 = 56 px
                               56 < MIN_STRIP_PX → PC DISABLED (trust CNC nominal)
    At calibrated ~8 000 px/mm: step_px = 2400  →  ov_x = 656 px → PC enabled ✓
    Calibrate px/mm: place two tiles side-by-side, measure matching-feature
    pixel distance, divide by step_mm (0.3).

Requirements
------------
    cupy-cuda12x >= 13.0   ← CUDA 12.8+ required for RTX 5060 Ti (Blackwell sm_120)
                              pip install cupy-cuda12x
    numpy, opencv-python

Integration (3 changes in scanner.py)
--------------------------------------
    # 1. At top of scanner.py:
    from hardware.stitcher import Stitcher

    # 2. In run_scan(), after the scan_started event:
    self.stitcher = Stitcher(
        pixels_per_mm=NOMINAL_PIXELS_PER_MM,
        step_mm=scan_step_mm,
    )

    # 3. In the per-tile loop, right after `with open(tile_path,'wb')`:
    await self.stitcher.on_tile_captured(row, col, x_mm, y_mm, filename)

Integration (main.py — live preview endpoint)
----------------------------------------------
    @app.get("/api/stitch/preview")
    async def stitch_preview():
        s = scanner_routine.stitcher
        if s is None:
            return JSONResponse({"error": "no active stitch"}, status_code=404)
        loop = asyncio.get_event_loop()
        jpeg = await loop.run_in_executor(None, s.generate_preview)
        if jpeg is None:
            return JSONResponse({"error": "no tiles yet"}, status_code=404)
        from fastapi.responses import Response
        return Response(content=jpeg, media_type="image/jpeg")
"""

from __future__ import annotations
import os, asyncio, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np

# ── GPU ───────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp   = None
    _GPU = False

log = logging.getLogger("stitcher")

# ── Constants ─────────────────────────────────────────────────────────────────
EDGE_CROP    = 500    # pixels discarded from each edge to remove lens artefacts

# Cache: 130 cropped-grayscale tiles ≈ 130 × (3056×2040×1B) ≈ 810 MB RAM.
# Rule of thumb: keep ≥ (max_cols_per_row + 1) tiles so the row-above tile is
# always cached when we start phase-correlating the next row.
# 130 comfortably covers any small skin-biopsy grid at 0.3 mm step.
CACHE_LIMIT  = 130

MIN_STRIP_PX = 150    # minimum overlap strip width to attempt phase correlation
MIN_CONF     = 0.05   # below this, edge weight → 0; tile stays at CNC nominal
K_EDGE       = 1.0
K_ANCHOR     = 0.05   # raise toward 0.3 if PC is unreliable; lower toward 0.01
                      # to let optics fully override CNC once you have good overlap
RELAX_N      = 50
RELAX_LR     = 0.4

# Thumbnail dimensions written by scanner.py
_THUMB_W = 160
_THUMB_H = 120


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileNode:
    col:      int
    row:      int
    filename: str                # absolute path to the saved full-resolution tile JPEG
    nominal:  np.ndarray         # [px_x, px_y] — effective tile top-left, float64
    offset:   np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    @property
    def position(self) -> np.ndarray:
        """Current refined top-left of the effective (cropped) tile in mosaic pixels."""
        return self.nominal + self.offset


@dataclass
class TileEdge:
    idx_a:          int          # left / top tile
    idx_b:          int          # right / bottom tile
    measured_shift: np.ndarray   # position_b − position_a, shape (2,)
    confidence:     float        # PC peak SNR normalised to [0, 1]
    direction:      str          # 'h' or 'v'

    @property
    def weight(self) -> float:
        return K_EDGE * max(self.confidence - MIN_CONF, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase correlation
# ─────────────────────────────────────────────────────────────────────────────

def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _phase_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Normalised phase correlation between two same-shape grayscale patches.
    Uses a Hann window to suppress spectral leakage.

    Convention (numpy irfft2 peak semantics):
        peak (ry, rx)  ⟹  b[y, x] ≈ a[y − ry, x − rx]
        positive rx  →  patch b is rx pixels to the RIGHT of patch a.

    Sign derivation for horizontal overlap strips (A = left tile, B = right)
    ─────────────────────────────────────────────────────────────────────────
    strip_a = rightmost ov_x cols of effective A  (world x: [xa + eff_w − ov_x, xa + eff_w])
    strip_b = leftmost  ov_x cols of effective B  (world x: [xb,  xb + ov_x])
    At nominal (xb = xa + step_px = xa + eff_w − ov_x) both strips cover the same tissue
    → expected PC peak = (0, 0).

    If B is δ px further right than nominal:
        strip_b[y, q] ≈ strip_a[y, q + δ]  ≡  a[y − 0, q − (−δ)]
        ⟹  PC peak  rx = −δ
        ⟹  actual B−A x-displacement = step_px + δ = step_px − rx         ✓

    Returns  (rx, ry, confidence)  with confidence ∈ [0, 1].
    Empirically: dense H&E tissue ≈ 0.25–1.0; blank glass < 0.06.
    """
    win = _hann2d(*a.shape[:2])
    aw  = a.astype(np.float32) * win
    bw  = b.astype(np.float32) * win

    if _GPU:
        fa, fb = cp.asarray(aw), cp.asarray(bw)
        A, B   = cp.fft.rfft2(fa), cp.fft.rfft2(fb)
        R      = A * cp.conj(B);  R /= (cp.abs(R) + 1e-8)
        r      = cp.fft.irfft2(R, s=a.shape[:2])
        flat   = int(cp.argmax(r))
        snr    = float(cp.max(r) / (cp.mean(cp.abs(r)) + 1e-8))
    else:
        A, B   = np.fft.rfft2(aw), np.fft.rfft2(bw)
        R      = A * np.conj(B);  R /= (np.abs(R) + 1e-8)
        r      = np.fft.irfft2(R, s=a.shape[:2])
        flat   = int(np.argmax(r))
        snr    = float(np.max(r) / (np.mean(np.abs(r)) + 1e-8))

    h, w   = a.shape[:2]
    ry, rx = divmod(flat, w)
    if ry > h // 2: ry -= h
    if rx > w // 2: rx -= w

    return float(rx), float(ry), float(np.clip(snr / 80.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# LRU image cache  — stores CROPPED grayscale tiles
# ─────────────────────────────────────────────────────────────────────────────

class _ImageCache:
    """
    Loads tiles as grayscale and immediately crops EDGE_CROP pixels from all
    four sides before caching.  This halves memory vs caching the full tile:
      full:    4056 × 3040 × 1B ≈ 12.3 MB / tile
      cropped: 3056 × 2040 × 1B ≈  6.2 MB / tile  →  130 tiles ≈ 810 MB RAM
    """

    def __init__(self, maxsize: int = CACHE_LIMIT, crop_px: int = EDGE_CROP):
        self._d:    Dict[str, np.ndarray] = {}
        self._max   = maxsize
        self._crop  = crop_px

    def get(self, path: str) -> Optional[np.ndarray]:
        if path in self._d:
            self._d[path] = self._d.pop(path)   # bump to end (LRU)
            return self._d[path]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        if self._crop > 0:
            c = self._crop
            img = img[c:-c, c:-c]               # store only the effective region

        while len(self._d) >= self._max:
            self._d.pop(next(iter(self._d)))    # evict oldest

        self._d[path] = img
        return img


# ─────────────────────────────────────────────────────────────────────────────
# TileMesh  — the registration graph
# ─────────────────────────────────────────────────────────────────────────────

class TileMesh:
    def __init__(self):
        self.nodes: List[TileNode] = []
        self.edges: List[TileEdge] = []
        self._idx:  Dict[Tuple[int, int], int] = {}

    def add_node(self, n: TileNode) -> int:
        i = len(self.nodes)
        self.nodes.append(n)
        self._idx[(n.col, n.row)] = i
        return i

    def node_at(self, col: int, row: int) -> Optional[int]:
        return self._idx.get((col, row))

    def add_edge(self, e: TileEdge):
        self.edges.append(e)

    def positions(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        return {
            (n.col, n.row): (float(n.position[0]), float(n.position[1]))
            for n in self.nodes
        }


# ─────────────────────────────────────────────────────────────────────────────
# Jacobi spring-mesh relaxation
# ─────────────────────────────────────────────────────────────────────────────

def relax(mesh: TileMesh, n: int = RELAX_N, lr: float = RELAX_LR) -> float:
    """
    Minimise the spring-mesh energy via Jacobi gradient descent:

        E = Σ_edges  (w_ij / 2) · ‖(p_j − p_i) − s_ij‖²
          + Σ_nodes  (K_ANCHOR / 2) · ‖offset_i‖²

    Edge spring:   pulls each tile pair toward their phase-correlation measurement.
    Anchor spring: pulls each offset back toward zero (CNC nominal position).

    Warm-startable — offsets carried over from the previous call mean only
    the new tile's neighbourhood needs to re-settle, not the whole mesh.

    Returns mean |Δoffset| in pixels as a convergence indicator.
    """
    N = len(mesh.nodes)
    if N == 0:
        return 0.0

    pos = np.array([nd.position for nd in mesh.nodes], dtype=np.float64)  # (N, 2)
    off = np.array([nd.offset   for nd in mesh.nodes], dtype=np.float64)

    for _ in range(n):
        forces = np.zeros_like(pos)
        for e in mesh.edges:
            w = e.weight
            if w == 0.0:
                continue
            # err = how much the current gap deviates from the measured shift
            # ∂E/∂p_a = −w·err  →  gradient-descent force on a = +w·err
            # ∂E/∂p_b = +w·err  →  gradient-descent force on b = −w·err
            err = (pos[e.idx_b] - pos[e.idx_a]) - e.measured_shift
            forces[e.idx_a] +=  w * err
            forces[e.idx_b] -= w * err

        forces -= K_ANCHOR * off   # anchor: ∂E/∂offset = K_ANCHOR·offset
        off   += lr * forces
        pos    = np.array([nd.nominal for nd in mesh.nodes]) + off

    delta = float(np.mean(np.abs(lr * forces)))
    for i, nd in enumerate(mesh.nodes):
        nd.offset = off[i].copy()
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# Stitcher
# ─────────────────────────────────────────────────────────────────────────────

class Stitcher:
    """
    Incremental tile stitcher for the OpenWSI scanner.

    Call await on_tile_captured() after each tile is written to disk.
    get_positions() returns current refined mosaic coordinates at any time.
    generate_preview() composites a JPEG thumbnail mosaic for the browser.
    """

    def __init__(
        self,
        pixels_per_mm: float = 10_000.0,
        step_mm:       float = 0.3,
        step_y_mm: float = None,

        crop_px:       int   = EDGE_CROP,
        tile_dir:      str   = "/tmp/wsi_scan",
        tile_w:        int   = 4056,
        tile_h:        int   = 3040,
    ):
        self.ppm      = pixels_per_mm
        self.step_px  = step_mm * pixels_per_mm
        self.step_y_px = (step_y_mm if step_y_mm is not None else step_mm) * pixels_per_mm
        self.crop_px  = crop_px
        self.tile_dir = tile_dir
        self.tile_w   = tile_w
        self.tile_h   = tile_h

        # Effective (post-crop) tile dimensions
        self.eff_w = tile_w - 2 * crop_px   # 4056 − 1000 = 3056 px
        self.eff_h = tile_h - 2 * crop_px   # 3040 − 1000 = 2040 px

        # Overlap of adjacent effective tiles.
        # ov_x = 3056 − step_px   (positive = overlap, negative = gap)
        # At NOMINAL 10 000 px/mm:  ov_x =  56 px  → PC disabled (< MIN_STRIP_PX)
        # At calibrated ~8 000 px/mm: ov_x = 656 px → PC enabled ✓
        self.ov_x = int(self.eff_w - round(self.step_px))
        self.ov_y = int(self.eff_h - round(self.step_y_px))

        # Proportional crop for the 160×120 thumbnails saved by scanner.py
        self._tcrop_x = round(crop_px * _THUMB_W / tile_w)   # ≈ 20 px
        self._tcrop_y = round(crop_px * _THUMB_H / tile_h)   # ≈ 20 px
        self._teff_w  = _THUMB_W - 2 * self._tcrop_x         # ≈ 120 px
        self._teff_h  = _THUMB_H - 2 * self._tcrop_y         # ≈ 80  px

        self.mesh   = TileMesh()
        self._cache = _ImageCache(CACHE_LIMIT, crop_px)

        log.info(
            f"[Stitcher] step=({step_mm}/{step_y_mm or step_mm}) mm x/y  "
            f"({self.step_px:.0f}/{self.step_y_px:.0f} px)  "
            f"crop={crop_px} px  eff={self.eff_w}×{self.eff_h}  "
            f"overlap=({self.ov_x} px h, {self.ov_y} px v)  "
            f"cache={CACHE_LIMIT} tiles (~{CACHE_LIMIT * self.eff_w * self.eff_h // 1_000_000} MB)  "
            f"GPU={'CuPy ✓' if _GPU else 'numpy fallback'}"
        )
        if self.ov_x < MIN_STRIP_PX:
            log.warning(
                f"[Stitcher] ov_x={self.ov_x} px < {MIN_STRIP_PX} — "
                "horizontal PC disabled.  Calibrate pixels_per_mm from a FOV "
                "measurement: measure pixel distance between matching features "
                "in adjacent tiles, divide by step_mm (0.3)."
            )
        if self.ov_y < MIN_STRIP_PX:
            log.warning(
                f"[Stitcher] ov_y={self.ov_y} px < {MIN_STRIP_PX} — "
                "vertical PC disabled."
            )

        # Optional async callback — called with updated positions after each tile.
        # async def on_update(positions: Dict[Tuple[int,int], Tuple[float,float]])
        self.on_update: Optional[Callable] = None

    # ── Entry point ───────────────────────────────────────────────────────────

    async def on_tile_captured(
        self,
        row:      int,
        col:      int,
        x_mm:     float,
        y_mm:     float,
        filename: str,
    ):
        """
        Call this immediately after scanner.py writes a tile to disk.
        Registers the tile against its neighbours and relaxes the mesh,
        both in the thread-pool so the scan loop is never blocked.
        """
        path = os.path.join(self.tile_dir, filename)

        # Nominal = top-left of the effective (cropped) tile in mosaic pixel space.
        # step_px is the stage displacement in pixels, which is also the distance
        # between adjacent effective tile origins (crop offset cancels out).
        nominal = np.array(
            [col * self.step_px, row * self.step_y_px], dtype=np.float64  # ← step_y_px for Y
        )
        idx  = self.mesh.add_node(TileNode(col, row, path, nominal))
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, self._register, idx, col, row)
        delta = await loop.run_in_executor(None, relax, self.mesh)
        log.debug(f"[Stitcher] ({col},{row}) relaxed, mean Δ = {delta:.3f} px")

        if self.on_update:
            await self.on_update(self.mesh.positions())

    # ── Registration (runs in executor thread) ────────────────────────────────

    def _register(self, new_idx: int, col: int, row: int):
        # Left neighbour — exists when the row is scanned left-to-right (even rows)
        nb_left = self.mesh.node_at(col - 1, row)
        if nb_left is not None:
            self._add_edge(nb_left, new_idx, 'h')   # A=col-1 (left), B=col (right) ✓

        # Right neighbour — exists when the row is scanned right-to-left (odd rows).
        # Swap idx_a/idx_b so A is always the LEFT tile regardless of scan direction.
        nb_right = self.mesh.node_at(col + 1, row)
        if nb_right is not None:
            self._add_edge(new_idx, nb_right, 'h')  # A=col (left), B=col+1 (right) ✓

        # Top neighbour — sparse grid means this may not exist for many tiles;
        # those tiles fall back to the CNC nominal y-position via the anchor spring.
        nb_top = self.mesh.node_at(col, row - 1)
        if nb_top is not None:
            self._add_edge(nb_top, new_idx, 'v')

    def _add_edge(self, idx_a: int, idx_b: int, direction: str):
        """
        Phase-correlate the facing overlap strip between two neighbours
        and add a weighted spring edge to the mesh.

        Images from the cache are already cropped to the effective region.
        Strip extraction and sign convention follow the derivation in _phase_corr.
        """
        na = self.mesh.nodes[idx_a]
        nb = self.mesh.nodes[idx_b]
        ov = self.ov_x if direction == 'h' else self.ov_y

        # Not enough overlap — add a zero-weight nominal edge as a topology hint
        # so the anchor spring still keeps the mesh connected.
        if ov < MIN_STRIP_PX:
            shift = (
                np.array([self.step_px, 0.0]) if direction == 'h'
                else np.array([0.0, self.step_y_px])  # ← step_y_px for vertical
            )
            self.mesh.add_edge(TileEdge(idx_a, idx_b, shift, 0.0, direction))
            return

        img_a = self._cache.get(na.filename)
        img_b = self._cache.get(nb.filename)
        if img_a is None or img_b is None:
            log.warning(
                f"[Stitcher] image load failed: {na.filename} or {nb.filename}"
            )
            return

        # Extract the facing overlap strips from effective-region images
        if direction == 'h':
            sa, sb = img_a[:, -ov:],  img_b[:, :ov]    # right edge A ↔ left edge B
        else:
            sa, sb = img_a[-ov:, :],  img_b[:ov, :]    # bottom edge A ↔ top edge B

        rx, ry, conf = _phase_corr(sa, sb)

        # Full tile-to-tile measured shift (see sign derivation in _phase_corr docstring)
        measured_shift = (
            np.array([self.step_px - rx, -ry]) if direction == 'h'
            else np.array([-rx, self.step_y_px - ry])  # ← step_y_px for vertical
        )
        log.debug(
            f"[Stitcher] PC ({na.col},{na.row})→({nb.col},{nb.row})  "
            f"dir={direction}  residual=({-rx:.1f}, {-ry:.1f}) px  conf={conf:.3f}"
        )
        self.mesh.add_edge(TileEdge(idx_a, idx_b, measured_shift, conf, direction))

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_positions(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Current refined positions for all tiles.
        { (col, row): (px_x, px_y) } — effective tile top-left in mosaic pixels.
        """
        return self.mesh.positions()

    # ── Live preview ──────────────────────────────────────────────────────────

    def generate_preview(self, max_dim: int = 1024, quality: int = 75) -> Optional[bytes]:
        """
        Composite a downscaled JPEG preview using the 160×120 thumbnails
        saved by scanner.py, cropped proportionally to match the effective region.

        Effective thumbnail size: (~120 × ~80 px after removing ~20 px per side).
        Safe to run in an executor thread alongside the live scan.
        Returns JPEG bytes, or None if no tiles are registered yet.
        """
        pos = self.mesh.positions()
        if not pos:
            return None

        xs = [v[0] for v in pos.values()]
        ys = [v[1] for v in pos.values()]
        min_x, min_y = min(xs), min(ys)
        max_x = max(xs) + self.eff_w
        max_y = max(ys) + self.eff_h

        scale = max_dim / max(max_x - min_x, max_y - min_y, 1)
        out_w = max(1, int((max_x - min_x) * scale))
        out_h = max(1, int((max_y - min_y) * scale))
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        tcx, tcy = self._tcrop_x, self._tcrop_y

        for nd in self.mesh.nodes:
            tile_name = os.path.basename(nd.filename)  # "tile_r001_c004.jpg"
            thumb_path = os.path.join(
                self.tile_dir, "thumb_" + tile_name[len("tile_"):]  # "thumb_r001_c004.jpg"
            )
            thumb = cv2.imread(thumb_path)
            if thumb is None:
                continue

            # Crop thumbnail to effective region (removes lens-artefact border)
            thumb_eff = thumb[tcy : _THUMB_H - tcy, tcx : _THUMB_W - tcx]

            dx = int((nd.position[0] - min_x) * scale)
            dy = int((nd.position[1] - min_y) * scale)
            tw = max(1, int(self.eff_w * scale))
            th = max(1, int(self.eff_h * scale))
            ts = cv2.resize(thumb_eff, (tw, th), interpolation=cv2.INTER_AREA)

            x1, y1 = max(dx, 0),       max(dy, 0)
            x2, y2 = min(dx + tw, out_w), min(dy + th, out_h)
            if x2 <= x1 or y2 <= y1:
                continue
            canvas[y1:y2, x1:x2] = ts[y1 - dy : y2 - dy, x1 - dx : x2 - dx]

        _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
