"""
Autofocus Engine — Tenengrad-only
---------------------------------
Pure Tenengrad hill-climb autofocus for the Canon lens ring.

Strategy per tile (when AF is triggered):
  1. Keep illumination on-axis for focus measurement.
  2. Measure Tenengrad from the live stream.
  3. Probe both lens directions with a minimum ring step of 25.
  4. Hill-climb to the local Tenengrad peak.
  5. Apply backlash-aware final approach from a consistent direction.

The ring range guard and CNC Z recovery climb remain active. Calibration files
are retained for compatibility with the existing API and future model-guided AF.
"""

import asyncio
import json
import os
import numpy as np
import cv2
from datetime import datetime


# ─── Config dataclass (plain dict wrapper for type safety) ────────────────────
class _AFConfig:
    def __init__(self, d: dict):
        self.direction_sign            = d.get("direction_sign")
        self.initial_jump_cap          = int(d.get("initial_jump_cap", 200))
        self.fine_step                 = max(50, int(d.get("fine_step", 50)))
        self.backlash_steps            = int(d.get("backlash_steps", 16))
        self.backlash_dir              = int(d.get("backlash_approach_direction", 1))
        self.max_climb_iters           = int(d.get("max_climb_iters", 25))
        self.decline_threshold         = float(d.get("decline_threshold", 0.03))
        self.ring_position_limit       = int(d.get("ring_position_limit", 600))   # ±600 steps
        self.z_hill_step_mm            = float(d.get("z_hill_step_mm", 0.01))     # CNC Z step when ring limit hit
        self.max_z_travel_mm           = float(d.get("max_z_travel_mm", 0.05))    # hard safety cap on Z travel
        self.local_drop                = float(d.get("local_drop_threshold", 0.05))
        self.global_drop               = float(d.get("global_drop_threshold", 0.60))
        self.global_window             = int(d.get("global_history_window", 30))
        self.min_tissue                = float(d.get("min_tissue_fraction", 0.03))
        self.sweep_range               = int(d.get("sweep_range_steps", 300))
        self.sweep_step                = int(d.get("sweep_step_size", 15))


# ─── Main class ───────────────────────────────────────────────────────────────
class AutoFocus:
    CONFIG_PATH = "calibration/af_config.json"
    LUT_PATH    = "calibration/confidence_lut.json"

    def __init__(self):
        # Hardware references — set by main.py after hardware connect
        self.camera = None
        self.canon  = None
        self.led    = None
        self.cnc    = None

        self.config = self._load_config()
        self.lut    = self._load_lut()

    # ── Readiness ──────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    # ── Config / LUT loading ───────────────────────────────────────────────────
    def _load_config(self) -> _AFConfig:
        try:
            with open(self.CONFIG_PATH) as f:
                raw = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
            print(f"[AF] Config loaded from {self.CONFIG_PATH}")
            return _AFConfig(raw)
        except FileNotFoundError:
            print(f"[AF] {self.CONFIG_PATH} not found — using defaults.")
            return _AFConfig({})

    def _load_lut(self) -> dict:
        try:
            with open(self.LUT_PATH) as f:
                lut = json.load(f)
            calibrated = sum(1 for b in lut["bins"] if b["n_samples"] > 0)
            print(f"[AF] LUT loaded — {calibrated}/20 bins calibrated.")
            return lut
        except FileNotFoundError:
            print(f"[AF] {self.LUT_PATH} not found — LUT uncalibrated.")
            return {"bins": [{"lo": i*0.05, "hi": (i+1)*0.05,
                               "mean_steps": 0, "n_samples": 0}
                              for i in range(20)],
                    "direction_sign": None, "calibration_date": None}

    def _save_lut(self):
        os.makedirs("calibration", exist_ok=True)
        with open(self.LUT_PATH, "w") as f:
            json.dump(self.lut, f, indent=4)

    def _save_config(self):
        os.makedirs("calibration", exist_ok=True)
        existing = {}
        try:
            with open(self.CONFIG_PATH) as f:
                existing = json.load(f)
        except Exception:
            pass
        existing["direction_sign"] = self.config.direction_sign
        with open(self.CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=4)

    # ── Camera helpers ─────────────────────────────────────────────────────────
    async def _grab_fresh_frame(self, timeout: float = 2.0) -> bytes:
        self.camera.new_frame_event.clear()
        self.camera.latest_frame = b''
        try:
            await asyncio.wait_for(self.camera.new_frame_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print("[AF] Frame grab timeout.")
        return self.camera.latest_frame

    @staticmethod
    def tenengrad(gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    async def _tenengrad_from_stream(self) -> float:
        frame = await self._grab_fresh_frame()
        if not frame:
            return 0.0
        arr  = np.frombuffer(frame, np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return self.tenengrad(gray) if gray is not None else 0.0

    # ── Ring range guard + CNC Z hill-climb ───────────────────────────────────
    async def _recenter_ring(self, loop) -> float | None:
        """
        Fires when Canon ring travel exceeds ±ring_position_limit (600 steps).

        Actions:
          1. Snap the ring back to relative zero (canon.focus(-current_pos)).
          2. Run a CNC Z Tenengrad recovery search with 0.01 mm probing, a 0.02 mm
             fallback probe when neither side improves, and a 0.05 mm hard cap in
             the chosen direction.
          3. Follow with the normal Canon ring Tenengrad hunt.

        Returns:
            float  — the final Tenengrad after Z recovery and ring hunt
            None   — limit not reached, no action taken
        """
        pos = self.canon.position
        if abs(pos) < self.config.ring_position_limit:
            return None

        print(f"\n[AF] ⚠  Ring limit reached (pos={pos}).  "
              f"Snapping ring to 0 and running CNC Z hill-climb.")

        # 1. Snap ring to relative zero
        await loop.run_in_executor(None, self.canon.focus, -pos)
        self.canon.reset_position()
        await asyncio.sleep(0.1)

        await self.led.async_on_axis()
        await asyncio.sleep(0.45)

        # 2. CNC Z Tenengrad recovery
        T_z = await self._z_tenengrad_climb(loop)
        print(f"[AF] Z climb done. T={T_z:.1f}  ring_pos={self.canon.position}")

        # 3. Finish with the normal Canon ring focus hunt
        T_final = await self._tenengrad_climb(loop)
        print(f"[AF] Ring hunt after recenter done. T={T_final:.1f}  ring_pos={self.canon.position}")
        return T_final

    async def _z_tenengrad_climb(self, loop) -> float:
        """
        Tenengrad-guided CNC Z recovery search.

        Probe sequence:
          1. Probe +0.01 mm and -0.01 mm from the starting position.
          2. If neither improves Tenengrad, return to start and probe +0.02 mm
             and -0.02 mm.
          3. Once a direction is selected, continue in 0.01 mm steps.

        Hard safety cap: movement is limited to at most 0.05 mm from the starting
        position in the chosen direction.

        CNC is temporarily switched to G91 relative mode for the Z moves,
        then restored to G90 absolute mode for the continuing scan.
        """
        cfg           = self.config
        z_mm          = cfg.z_hill_step_mm
        max_travel_mm = min(cfg.max_z_travel_mm, 0.05)

        # Switch to relative mode so Z moves are incremental
        await loop.run_in_executor(None, self.cnc.send_and_wait_ok, "G91")
        try:
            current_offset_mm = 0.0

            async def z_move_and_measure(delta_mm: float) -> float:
                nonlocal current_offset_mm
                if abs(delta_mm) > 1e-9:
                    cmd = f"G0 Z{delta_mm:.4f} F150"
                    await loop.run_in_executor(None, self.cnc.send_and_wait_ok, cmd)
                    await loop.run_in_executor(None, self.cnc.wait_for_idle_blocking)
                    current_offset_mm += delta_mm
                await asyncio.sleep(0.5)
                return await self._tenengrad_from_stream()

            async def move_to_and_measure(target_offset_mm: float) -> float:
                return await z_move_and_measure(target_offset_mm - current_offset_mm)

            async def probe_offsets(step_mm: float) -> tuple[int | None, float]:
                """
                Probe both directions at the requested absolute offset from the
                starting position and return the improving direction, if any.
                """
                T_plus = await move_to_and_measure(+step_mm)
                T_minus = await move_to_and_measure(-step_mm)

                plus_improved = T_plus > T0
                minus_improved = T_minus > T0

                if plus_improved and (not minus_improved or T_plus >= T_minus):
                    await move_to_and_measure(+step_mm)
                    return +1, T_plus
                if minus_improved:
                    return -1, T_minus

                return None, T0

            T0 = await self._tenengrad_from_stream()

            direction, best_T = await probe_offsets(z_mm)
            if direction is None:
                await move_to_and_measure(0.0)
                direction, best_T = await probe_offsets(2 * z_mm)
                if direction is None:
                    await move_to_and_measure(0.0)
                    return T0

            climb_step_mm = direction * z_mm

            while True:
                next_offset_mm = current_offset_mm + climb_step_mm
                if abs(next_offset_mm) > max_travel_mm + 1e-9:
                    print(f"[AF] Z climb safety limit reached "
                          f"({abs(current_offset_mm) * 1000:.0f}µm / "
                          f"{max_travel_mm * 1000:.0f}µm max). Stopping.")
                    break

                T_new = await z_move_and_measure(climb_step_mm)
                if T_new < best_T * (1.0 - cfg.decline_threshold):
                    await z_move_and_measure(-climb_step_mm)
                    break
                best_T = T_new

            return best_T
        finally:
            # Restore absolute positioning for the scan
            await loop.run_in_executor(None, self.cnc.send_and_wait_ok, "G90")

    # ── Tenengrad hill-climb (Canon ring) ─────────────────────────────────────
    async def _tenengrad_climb(self, loop) -> float:
        """
        Under on-axis illumination, find the focus peak via small step probing.

        1. Probe +fine_step → compare to baseline.
        2. If better → continue that way. If worse → probe −fine_step.
        3. Climb until Tenengrad drops by more than decline_threshold.
        4. Backlash-aware final approach.

        Returns the best Tenengrad measured.
        """
        cfg = self.config
        fs  = cfg.fine_step

        async def step_and_measure(steps: int) -> float:
            await loop.run_in_executor(None, self.canon.focus, steps)
            await asyncio.sleep(0.5)
            return await self._tenengrad_from_stream()

        T0 = await self._tenengrad_from_stream()

        # ── Direction probe ──────────────────────────────────────────
        T_up = await step_and_measure(+fs)
        if T_up >= T0:
            direction = +1
            T_prev    = T_up
        else:
            # Undo +fs, try −fs
            T_down = await step_and_measure(-2 * fs)     # back past T0 position by −fs
            if T_down >= T0:
                direction = -1
                T_prev    = T_down
            else:
                # T0 was already best — restore and return
                await loop.run_in_executor(None, self.canon.focus, +fs)
                return T0

        # ── Climb ────────────────────────────────────────────────────
        best_T = T_prev
        for _ in range(cfg.max_climb_iters):
            T_new = await step_and_measure(direction * fs)
            if T_new < best_T * (1.0 - cfg.decline_threshold):
                # Overshot by one step — back up
                await loop.run_in_executor(None, self.canon.focus, -direction * fs)
                break
            best_T = T_new

        # ── Backlash correction (approach from backlash_dir consistently) ──
        if direction != cfg.backlash_dir:
            # We climbed in the opposite of preferred direction.
            # Overshoot by backlash_steps, then re-approach from backlash_dir.
            await loop.run_in_executor(None, self.canon.focus,
                                       -cfg.backlash_dir * cfg.backlash_steps)
            await asyncio.sleep(0.03)
            await loop.run_in_executor(None, self.canon.focus,
                                       +cfg.backlash_dir * cfg.backlash_steps)
            best_T = await self._tenengrad_from_stream()

        return best_T

    # ── Main AF entry point ────────────────────────────────────────────────────
    async def run(self, loop) -> float:
        """
        Full AF routine for one tile.
        Called by scanner.py when the local or global gate triggers.
        Returns the final on-axis Tenengrad (focus quality metric).
        """
        if not (self.camera and self.canon and self.led):
            print("[AF] run() called but hardware not bound — skipping.")
            return 0.0

        # Range guard: if ring has hit its travel limit, snap it to 0, do the
        # CNC Z recovery search, then finish with the normal ring hunt.
        z_climb_result = await self._recenter_ring(loop)
        if z_climb_result is not None:
            return z_climb_result

        try:
            # Tenengrad-only AF always runs under on-axis illumination.
            await self.led.async_on_axis()
            await asyncio.sleep(0.45)

            T_final = await self._tenengrad_climb(loop)
            print(f"[AF] Done. T_final={T_final:.1f}  ring_pos={self.canon.position}")
            return T_final
        finally:
            await self.led.async_on_axis()

    async def recover_low_global_focus(self, loop) -> float:
        """
        Recovery path for catastrophic global focus loss:
          1. Run a CNC Z Tenengrad recovery search with 0.01 mm probes, a
             0.02 mm fallback probe when neither side improves, and a 0.05 mm
             cap in the climb direction.
          2. Follow with the normal Canon ring Tenengrad correction.

        Returns the final Tenengrad after both recovery stages.
        """
        if not (self.camera and self.canon and self.led and self.cnc):
            print("[AF] recover_low_global_focus() called but hardware not bound.")
            return 0.0

        try:
            await self.led.async_on_axis()
            await asyncio.sleep(0.45)

            T_z = await self._z_tenengrad_climb(loop)
            print(f"[AF] Global focus recovery Z stage done. T={T_z:.1f}")

            T_final = await self._tenengrad_climb(loop)
            print(f"[AF] Global focus recovery ring stage done. T={T_final:.1f}")
            return T_final
        finally:
            await self.led.async_on_axis()

    # ── Focus sweep calibration ────────────────────────────────────────────────
    async def focus_sweep(self, loop, event_queue=None) -> dict:
        """
        Diagnostic Tenengrad sweep around the current position.
        The result shape is kept compatible with the existing calibration API.
        """
        if not (self.camera and self.canon and self.led):
            print("[AF] focus_sweep(): hardware not bound.")
            return {}

        cfg   = self.config
        steps = list(range(-cfg.sweep_range, cfg.sweep_range + 1, cfg.sweep_step))
        n     = len(steps)
        data  = []    # list of {ring_pos, prob, confidence, tenengrad}

        print(f"\n[AF] Starting Tenengrad sweep: ±{cfg.sweep_range} steps, "
              f"{cfg.sweep_step}/step → {n} positions")

        await self._recenter_ring(loop)
        await self.led.async_on_axis()
        await asyncio.sleep(0.45)

        # Move to start of sweep (most negative position)
        await loop.run_in_executor(None, self.canon.focus, -cfg.sweep_range)
        await asyncio.sleep(0.5)

        for i, rel_pos in enumerate(steps):
            if event_queue:
                await event_queue.put({
                    "event":    "calibration_sweep_progress",
                    "step":     i + 1,
                    "total":    n,
                    "ring_pos": rel_pos,
                })

            T = await self._tenengrad_from_stream()

            data.append({"ring_pos": rel_pos, "prob": 0.5,
                         "confidence": 0.0, "tenengrad": round(T, 2)})

            # Step to next position (except last)
            if i < n - 1:
                await loop.run_in_executor(None, self.canon.focus, cfg.sweep_step)
                await asyncio.sleep(0.5)

        # ── Return to sweep start ───────────────────────────────────────
        await loop.run_in_executor(None, self.canon.focus, -cfg.sweep_range)
        print("[AF] Sweep done. Analysing Tenengrad data...")

        # ── Find focus peak from Tenengrad ──────────────────────────────
        best_idx    = max(range(len(data)), key=lambda k: data[k]["tenengrad"])
        focus_pos   = data[best_idx]["ring_pos"]
        best_T      = data[best_idx]["tenengrad"]
        print(f"[AF] Focus peak at ring_pos={focus_pos}  Tenengrad={best_T:.1f}")

        new_bins = [{
            "lo": round(idx * 0.05, 2),
            "hi": round(idx * 0.05 + 0.05, 2),
            "mean_steps": 0,
            "n_samples": 0,
        } for idx in range(20)]

        self.lut = {
            "_comment":        "Model-guided LUT disabled in Tenengrad-only mode.",
            "version":         1,
            "n_bins":          20,
            "bin_width":       0.05,
            "direction_sign":  None,
            "calibration_date": datetime.now().isoformat(timespec="minutes"),
            "bins":            new_bins,
        }
        self._save_lut()

        result = {
            "data":       data,
            "focus_pos":  focus_pos,
            "best_T":     best_T,
            "sign":       None,
            "lut_bins":   new_bins,
        }

        if event_queue:
            await event_queue.put({"event": "calibration_sweep_result", **result})

        print("[AF] Tenengrad sweep complete.")
        return result
