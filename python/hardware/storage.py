"""
Storage configuration — single source of truth for every save path.
--------------------------------------------------------------------
Phase 1 of the SSD-relocation work. All acquisition output now lives under a
single user-selected STORAGE_ROOT (an SSD mounted on the Uno Q's USB hub),
resolved ONCE at process startup. No module elsewhere should hardcode a
'/tmp/wsi_*' path again — import the constants from here instead.

Layout under STORAGE_ROOT:

    <root>/
    ├── tmp/
    │   ├── prescan/     coarse prescan thumbnails + tissue_map.json   (served at /tmp_cache)
    │   └── scan/        live high-res working tiles + thumbnails + logs (served at /tmp_scan)
    ├── calibration/     FFC dark.npy / flat.npy  (persistent, decoupled from scan output)
    └── scan_exports/
        └── scan_<ts>/   exported tiles + manifest + TileConfiguration

Path resolution priority at startup:
    1. persisted settings file (written by the browser 'set storage' action)
    2. WSI_STORAGE_ROOT environment variable
    3. a LOCAL fallback dir  → flagged configured=False, scans refused until set

Safety: the unmounted-SSD footgun
    If the SSD is not mounted, its mountpoint still exists as an empty directory
    on the 32 GB eMMC, so writes SUCCEED SILENTLY onto the eMMC — the exact thing
    we are trying to avoid. We detect this by comparing the st_dev of the storage
    root against the st_dev of '/'. A properly mounted SSD lives on a different
    device than the root filesystem; if they match, the SSD is (almost certainly)
    not mounted and we refuse to run. Set allow_same_device=true in the settings
    file only if you intentionally want to store on the same disk as the OS.
"""

from __future__ import annotations

import json
import os
import shutil

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
# PROJECT_ROOT = parent of the 'hardware' package (where main.py lives).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Persisted settings live NEXT TO THE CODE, never on the SSD — we need this file
# to *find* the SSD, so it cannot depend on the SSD being mounted.
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "wsi_settings.json")

# Used only when no valid root is configured yet, so the app can still boot.
FALLBACK_ROOT = os.path.join(PROJECT_ROOT, "_storage_unconfigured")

ENV_VAR = "WSI_STORAGE_ROOT"

# Minimum free space (bytes) for is_ready(). This is a floor sanity check only;
# the real per-scan estimate (tiles × size × safety) is enforced at scan start.
MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024   # 1 GiB


class Storage:
    """Resolves and holds every save path. Instantiated once (see module singleton)."""

    def __init__(self):
        self.root: str = FALLBACK_ROOT
        self.configured: bool = False        # True only when a real root was resolved
        self.allow_same_device: bool = False # override for the mount guard
        self._resolve()

    # --------------------------------------------------------------
    # Resolution
    # --------------------------------------------------------------
    def _resolve(self) -> None:
        settings = self._load_settings()
        root = None
        if settings.get("storage_root"):
            root = settings["storage_root"]
        elif os.environ.get(ENV_VAR):
            root = os.environ[ENV_VAR]

        self.allow_same_device = bool(settings.get("allow_same_device", False))

        if root:
            self.root = os.path.abspath(root)
            self.configured = True
        else:
            self.root = FALLBACK_ROOT
            self.configured = False

        # Create the tree eagerly. If the SSD is unmounted this creates a few tiny
        # empty dirs on the eMMC — harmless; the mount guard will refuse writes.
        try:
            self.ensure_dirs()
        except OSError as e:
            print(f"[Storage] Could not create dirs under {self.root}: {e}")

    def _load_settings(self) -> dict:
        try:
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Storage] settings file unreadable ({e}); using defaults.")
            return {}

    # --------------------------------------------------------------
    # Derived paths  (all computed from self.root — never hardcode elsewhere)
    # --------------------------------------------------------------
    @property
    def tmp_dir(self) -> str:
        return os.path.join(self.root, "tmp")

    @property
    def prescan_dir(self) -> str:
        """Replaces /tmp/wsi_prescan — served at /tmp_cache."""
        return os.path.join(self.root, "tmp", "prescan")

    @property
    def scan_live_dir(self) -> str:
        """Replaces /tmp/wsi_scan — live working tiles/thumbs/logs, served at /tmp_scan."""
        return os.path.join(self.root, "tmp", "scan")

    @property
    def calib_dir(self) -> str:
        """Replaces /tmp/wsi_scan/calib — now persistent and decoupled from scan output."""
        return os.path.join(self.root, "calibration")

    @property
    def exports_root(self) -> str:
        """Replaces PROJECT_ROOT/scan_exports — timestamped bundles created underneath."""
        return os.path.join(self.root, "scan_exports")

    def ensure_dirs(self) -> None:
        for d in (self.prescan_dir, self.scan_live_dir, self.calib_dir, self.exports_root):
            os.makedirs(d, exist_ok=True)

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    def on_separate_device(self, path: str | None = None) -> bool:
        """True if `path` sits on a different device than '/' (i.e. an actual mount)."""
        p = path or self.root
        try:
            return os.stat(p).st_dev != os.stat("/").st_dev
        except OSError:
            return False

    def is_writable(self, path: str | None = None) -> bool:
        p = path or self.root
        probe = os.path.join(p, ".wsi_write_test")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return True
        except OSError:
            return False

    def free_bytes(self, path: str | None = None) -> int:
        p = path or self.root
        try:
            return shutil.disk_usage(p).free
        except OSError:
            return 0

    def mount_ok(self) -> bool:
        """The footgun guard: real mount unless explicitly overridden."""
        return self.allow_same_device or self.on_separate_device()

    def is_ready(self) -> bool:
        """All conditions required before a scan / FFC save may write to the root."""
        return (
            self.configured
            and self.mount_ok()
            and self.is_writable()
            and self.free_bytes() >= MIN_FREE_BYTES
        )

    def status(self) -> dict:
        """Full state for the browser storage panel."""
        return {
            "configured": self.configured,
            "root": self.root,
            "mount_ok": self.mount_ok(),
            "on_separate_device": self.on_separate_device(),
            "allow_same_device": self.allow_same_device,
            "writable": self.is_writable(),
            "free_bytes": self.free_bytes(),
            "free_gib": round(self.free_bytes() / (1024 ** 3), 2),
            "min_free_gib": round(MIN_FREE_BYTES / (1024 ** 3), 2),
            "ready": self.is_ready(),
            "settings_path": SETTINGS_PATH,
            "requires_restart": False,
        }

    # --------------------------------------------------------------
    # Configuration (called by POST /api/storage/config)
    # --------------------------------------------------------------
    def validate_candidate(self, candidate: str, allow_same_device: bool = False) -> dict:
        """
        Check a proposed root WITHOUT persisting it. Returns {ok, reason, ...}.
        The path must exist (we do not create arbitrary user paths blindly here),
        be a directory, be writable, sit on a separate device (unless overridden),
        and have enough free space.
        """
        candidate = os.path.abspath(os.path.expanduser(candidate.strip()))
        result = {"ok": False, "candidate": candidate, "reason": None}

        if not candidate or candidate == "/":
            result["reason"] = "Empty or root-filesystem path is not allowed."
            return result
        if not os.path.isdir(candidate):
            print(candidate)
            result["reason"] = f"Path does not exist or is not a directory: {candidate}"
            return result
        if not self.is_writable(candidate):
            result["reason"] = f"Path is not writable by the service user: {candidate}"
            return result

        sep = self.on_separate_device(candidate)
        if not sep and not allow_same_device:
            result["reason"] = (
                "Path is on the same device as the OS root — the SSD looks unmounted "
                "(writes would fill the eMMC). Mount the SSD, or set allow_same_device."
            )
            return result

        free = self.free_bytes(candidate)
        if free < MIN_FREE_BYTES:
            result["reason"] = (
                f"Only {round(free / 1024**3, 2)} GiB free; need at least "
                f"{round(MIN_FREE_BYTES / 1024**3, 2)} GiB."
            )
            return result

        result.update({
            "ok": True,
            "on_separate_device": sep,
            "free_gib": round(free / (1024 ** 3), 2),
        })
        return result

    def set_root(self, candidate: str, allow_same_device: bool = False) -> dict:
        """
        Validate + persist a new root to the settings file. Does NOT rebind the
        live process (static mounts + FFC load are bound at startup), so the
        caller must tell the user to restart the service for it to take effect.
        """
        check = self.validate_candidate(candidate, allow_same_device=allow_same_device)
        if not check["ok"]:
            return check

        settings = self._load_settings()
        settings["storage_root"] = os.path.abspath(os.path.expanduser(candidate.strip()))
        settings["allow_same_device"] = bool(allow_same_device)
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=2)
        except OSError as e:
            return {"ok": False, "reason": f"Failed to write settings file: {e}"}

        check["persisted"] = True
        check["requires_restart"] = True
        return check

    # --------------------------------------------------------------
    # tmp maintenance (called by POST /api/storage/tmp/clear)
    # --------------------------------------------------------------
    def clear_tmp(self, scope: str = "scan_thumbs") -> dict:
        """
        Delete files under tmp/. Path-guarded: can only ever touch dirs *inside*
        self.tmp_dir. Scopes:
            'scan_thumbs' → tmp/scan   (live scan working files + thumbnails)
            'prescan'     → tmp/prescan (prescan thumbs + tissue_map.json)
            'all'         → both of the above
        Recreates the emptied dirs afterward. The caller is responsible for the
        active-scan guard and the tissue-map-wipe warning.
        """
        targets_map = {
            "scan_thumbs": [self.scan_live_dir],
            "prescan": [self.prescan_dir],
            "all": [self.scan_live_dir, self.prescan_dir],
        }
        if scope not in targets_map:
            return {"ok": False, "reason": f"Unknown scope '{scope}'."}

        tmp_root = os.path.realpath(self.tmp_dir)
        before = self.free_bytes()
        removed = 0

        for target in targets_map[scope]:
            real = os.path.realpath(target)
            # Hard guard: refuse anything that is not strictly inside tmp/.
            if os.path.commonpath([real, tmp_root]) != tmp_root or real == tmp_root:
                return {"ok": False, "reason": f"Refusing to clear path outside tmp/: {real}"}
            if not os.path.isdir(real):
                continue
            for name in os.listdir(real):
                p = os.path.join(real, name)
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                    removed += 1
                except OSError as e:
                    print(f"[Storage] clear_tmp could not remove {p}: {e}")

        self.ensure_dirs()  # recreate emptied subdirs
        after = self.free_bytes()
        return {
            "ok": True,
            "scope": scope,
            "removed_entries": removed,
            "freed_bytes": max(after - before, 0),
            "freed_gib": round(max(after - before, 0) / (1024 ** 3), 2),
            "free_gib": round(after / (1024 ** 3), 2),
        }


# Module singleton — import THIS everywhere.
storage = Storage()