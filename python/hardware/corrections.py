import os
import numpy as np
import cv2

CALIB_DIR = "/tmp/wsi_scan/calib"

class FlatFieldCorrector:

    def __init__(self, calib_dir: str = CALIB_DIR):
        self.calib_dir = calib_dir
        self.dark:  np.ndarray | None = None   # float32, H×W×3
        self.flat:  np.ndarray | None = None   # float32, H×W×3
        self._gain: np.ndarray | None = None   # precomputed per-pixel, H×W×3
        self._ready = False

    # ── Stacking ──────────────────────────────────────────────────────
    @staticmethod
    def _decode_snap(snap_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(snap_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _median_stack(frames: list[np.ndarray]) -> np.ndarray:
        return np.median(
            np.stack([f.astype(np.float32) for f in frames], axis=0),
            axis=0
        )

    # ── Calibration setters ────────────────────────────────────────────
    def set_dark(self, frames: list[np.ndarray]) -> None:
        self.dark = self._median_stack(frames)

    def set_flat(self, frames: list[np.ndarray]) -> None:
        self.flat = self._median_stack(frames)
        self._compile()

    def _compile(self, eps: float = 1.0) -> None:
        """Precompute gain map once. Per-tile apply() becomes a single FMA."""
        denom = np.clip(self.flat - self.dark, eps, None)          # H×W×3
        channel_mean = denom.mean(axis=(0, 1), keepdims=True)      # 1×1×3
        self._gain = channel_mean / denom                           # H×W×3
        self._ready = True

    # ── Per-tile correction ────────────────────────────────────────────
    def apply(self, bgr: np.ndarray) -> np.ndarray:
        """Correct a full-resolution BGR tile. Fail-open if not calibrated."""
        if not self._ready:
            return bgr
        corrected = (bgr.astype(np.float32) - self.dark) * self._gain
        return np.clip(corrected, 0, 255).astype(np.uint8)

    # ── Validation ────────────────────────────────────────────────────
    def flatness_pct(self) -> float | None:
        """
        Measure intensity uniformity of the corrected flat frame itself.
        Apply correction to flat, compute std/mean of luminance.
        Lower = flatter. <5% is good. Returns None if not calibrated.
        """
        if not self._ready:
            return None
        corrected = self.apply(self.flat.astype(np.uint8))
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.std(gray) / np.mean(gray) * 100)

    # ── Persistence ───────────────────────────────────────────────────
    def save(self) -> None:
        os.makedirs(self.calib_dir, exist_ok=True)
        np.save(os.path.join(self.calib_dir, "dark.npy"), self.dark)
        np.save(os.path.join(self.calib_dir, "flat.npy"), self.flat)

    def load(self) -> bool:
        d = os.path.join(self.calib_dir, "dark.npy")
        f = os.path.join(self.calib_dir, "flat.npy")
        if not (os.path.exists(d) and os.path.exists(f)):
            return False
        self.dark = np.load(d).astype(np.float32)
        self.flat = np.load(f).astype(np.float32)
        self._compile()
        return True