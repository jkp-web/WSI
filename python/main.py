"""
WSI Scanner — FastAPI Server
-----------------------------
Runs on the PC (Ryzen / CachyOS).

Phase 2 additions:
  - AutoFocus          : FCFNN-directed Tenengrad hill-climb engine
  - /api/scan/resume   : POST — resume after manual focus pause
  - /api/calibration/sweep     : POST — run focus sweep, build confidence LUT
  - /api/calibration/af_status : GET  — return current calibration state
"""

import asyncio
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from camera_client import CameraClient
from hardware.cnc_controller import CNCController
from hardware.canon_lens import CanonLens
from hardware.led_controller import LedController
from hardware.autofocus import AutoFocus
from hardware.scanner import ScannerRoutine
from hardware.corrections import FlatFieldCorrector

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(title="WSI Scanner")


current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

# Force create the directory inside the container if it's missing
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

os.makedirs("/tmp/wsi_prescan", exist_ok=True)
os.makedirs("/tmp/wsi_scan",    exist_ok=True)

app.mount("/tmp_cache", StaticFiles(directory="/tmp/wsi_prescan"), name="tmp_cache")
app.mount("/tmp_scan",  StaticFiles(directory="/tmp/wsi_scan"),    name="tmp_scan")

# ------------------------------------------------------------------
# Hardware globals  (connected lazily via /api/hardware/connect)
# ------------------------------------------------------------------
camera    = CameraClient(host='192.168.0.4', port=8000)
cnc       = CNCController(port='/dev/ttyUSB1', baud_rate=115200)
canon:    CanonLens  | None = None
led:      LedController | None = None
autofocus: AutoFocus | None = None

# AutoFocus is loaded at startup — model weights are on disk, hardware refs
# are injected after /api/hardware/connect.
try:
    autofocus = AutoFocus()
except Exception as e:
    print(f"[Main] AutoFocus init failed: {e}")

scanner = ScannerRoutine(cnc=cnc, camera=camera)

# FFC — load calibration from disk if it exists; injected into scanner after load
ffc = FlatFieldCorrector()
_ffc_loaded = ffc.load()
if _ffc_loaded:
    scanner.ffc = ffc
    print("[Main] FFC calibration loaded from disk.")
else:
    print("[Main] No FFC calibration found — running uncorrected until calibrated.")

# Staging buffer for dark frames captured in step 1 of FFC calibration
_ffc_dark_frames: list = []

# Active WebSocket event clients
event_clients: list[WebSocket] = []

# ------------------------------------------------------------------
# Request/response models
# ------------------------------------------------------------------
class JogPayload(BaseModel):
    axis:     str
    distance: float

class LensPayload(BaseModel):
    direction: str
    steps:     int

class GainPayload(BaseModel):
    red:  float
    blue: float

class LedModePayload(BaseModel):
    mode: int   # 1 = on-axis (brightfield), 2 = off-axis (AF)

class ConnectHardwarePayload(BaseModel):
    cnc_port:    str = "/dev/ttyUSB3"
    canon_port:  str = "/dev/ttyUSB0"
    led_port:    str = "/dev/ttyUSB1"   # Adjust to match your LED Arduino

class HighResScanPayload(BaseModel):
    scan_whole_rows: bool = False

class ReimageSelectionPayload(BaseModel):
    filenames:  list[str] = []
    select_all: bool      = False

# ------------------------------------------------------------------
# WebSocket event broadcaster  (scanner → browser)
# ------------------------------------------------------------------
async def broadcast_events():
    while True:
        event_data = await scanner.event_queue.get()

        # Serialise ONCE outside the client loop.
        # A TypeError here (e.g. numpy scalar in payload) must NOT remove clients —
        # it should log and drop the event, leaving all connections intact.
        try:
            json_str = json.dumps(event_data)
        except TypeError as e:
            print(f"[Events] JSON serialisation error (event dropped): {e}  "
                  f"data keys: {list(event_data.keys()) if isinstance(event_data, dict) else '?'}")
            continue

        dead = []
        for client in event_clients:
            try:
                await client.send_text(json_str)
            except Exception:
                dead.append(client)
        for c in dead:
            event_clients.remove(c)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_events())

@app.websocket("/ws/events")
async def events_stream(ws: WebSocket):
    await ws.accept()
    event_clients.append(ws)
    try:
        while True:
            await ws.receive_text()   # Keep-alive
    except WebSocketDisconnect:
        if ws in event_clients:
            event_clients.remove(ws)

# ------------------------------------------------------------------
# Static / root
# ------------------------------------------------------------------
@app.get("/")
async def root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "static", "index.html")
    return FileResponse(index_path)

@app.get("/favicon.ico")
async def favicon():
    # Silence browser favicon requests — no icon file needed for this tool
    from fastapi.responses import Response
    return Response(status_code=204)

# ------------------------------------------------------------------
# Camera
# ------------------------------------------------------------------
@app.post("/api/camera/connect")
async def connect_camera():
    success = await camera.connect()
    return {"status": "success" if success else "error"}

@app.post("/api/camera/gains")
async def update_gains(payload: GainPayload):
    if camera.connected and camera.writer:
        cmd = f"GAIN {payload.red:.2f} {payload.blue:.2f}"
        camera.writer.write(cmd.encode('utf-8'))
        await camera.writer.drain()
        return {"status": "ok"}
    return {"status": "error", "message": "Camera not connected"}

# ------------------------------------------------------------------
# Hardware connection
# ------------------------------------------------------------------
@app.post("/api/hardware/connect")
async def connect_hardware(payload: ConnectHardwarePayload = ConnectHardwarePayload()):
    # Added cnc to globals if it's managed globally like led
    global cnc, canon, led 

    # 1. Connect CNC
    cnc.port = payload.cnc_port
    cnc_ok = cnc.connect()

    # 2. Connect Canon Lens (Commented out / Disabled)
    """
    try:
        canon    = CanonLens(port=payload.canon_port)
        canon_ok = True
    except Exception as e:
        print(f"[API] Canon lens failed: {e}")
        canon_ok = False
    """
    canon_ok = False  # Hardcoded fallback since the block above is disabled

    # 3. Connect LED Controller
    try:
        led    = LedController(port=payload.led_port)
        led_ok = led.connect()
    except Exception as e:
        print(f"[API] LED controller failed: {e}")
        led_ok = False

    # ── Inject hardware refs into scanner and autofocus ──────────────────
    scanner.led = led
    
    """
    scanner.canon     = canon
    scanner.autofocus = autofocus

    if autofocus:
        autofocus.camera = camera
        autofocus.canon  = canon
        autofocus.led    = led
        autofocus.cnc    = cnc
    """
    
    # Safely inject LED into autofocus only if autofocus exists
    if autofocus:
        autofocus.led = led

    # 4. Return status
    return {
        "cnc": cnc_ok, 
        "canon": canon_ok, 
        "led": led_ok,
        "autofocus": autofocus.is_ready if autofocus else False,
        "af_calibrated": autofocus.is_calibrated if autofocus else False
    }

# ------------------------------------------------------------------
# Manual jogging
# ------------------------------------------------------------------
@app.post("/api/jog/cnc")
async def jog_cnc(payload: JogPayload):
    feedrate = 800 if payload.axis in ['X', 'Y'] else 200
    loop = asyncio.get_running_loop()
    moved = await loop.run_in_executor(
        None, cnc.manual_jog, payload.axis, payload.distance, feedrate
    )
    if moved:
        return {"status": "ok"}
    return {"status": "error", "message": "CNC jog failed"}

@app.post("/api/jog/lens")
async def jog_lens(payload: LensPayload):
    if canon:
        steps = payload.steps if payload.direction == "near" else -payload.steps
        canon.focus(steps)
        return {"status": "ok", "steps_moved": steps}
    return {"status": "error", "message": "Lens not connected"}

# ------------------------------------------------------------------
# LED illumination mode
# ------------------------------------------------------------------
@app.post("/api/led/mode")
async def set_led_mode(payload: LedModePayload):
    """Switch illumination mode: 1 = on-axis brightfield, 2 = off-axis AF."""
    if led:
        success = led.set_mode(payload.mode)
        label   = "on-axis" if payload.mode == 1 else "off-axis"
        return {"status": "ok" if success else "error", "mode": label}
    return {"status": "error", "message": "LED controller not connected"}

# ------------------------------------------------------------------
# Prescan
# ------------------------------------------------------------------
@app.post("/api/scan/prescan")
async def start_prescan(bg: BackgroundTasks):
    if scanner.is_scanning:
        return {"status": "error", "message": "Scan already running"}
    bg.add_task(scanner.run_prescan)
    return {"status": "started"}

# ------------------------------------------------------------------
# High-resolution scan
# ------------------------------------------------------------------
@app.post("/api/scan/highres")
async def start_highres_scan(
    bg: BackgroundTasks,
    payload: HighResScanPayload = HighResScanPayload(),
):
    """
    Confirm the scan origin (current CNC position becomes 0,0) and start
    the high-resolution tile acquisition in the background.

    The user must have manually returned the CNC to the scan origin
    (same physical position as the start of prescan) before calling this.
    """
    if scanner.is_scanning:
        return {"status": "error", "message": "Scan already running"}

    # Set current CNC position as the absolute origin for this scan
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, cnc.set_origin)

    bg.add_task(scanner.run_scan, scan_whole_rows=payload.scan_whole_rows)
    return {"status": "started"}

@app.post("/api/scan/stop")
async def stop_scan():
    scanner.stop()
    return {"status": "ok", "message": "Stop signal sent"}


@app.get("/api/stitch/preview")
async def stitch_preview():
    """
    Returns a JPEG composite of all tiles registered so far,
    using the current spring-mesh positions.
    Poll this during a scan to watch the mosaic grow live.
    """
    if scanner.stitcher is None:
        return JSONResponse({"error": "no active stitch"}, status_code=404)
    from fastapi.responses import Response
    loop = asyncio.get_running_loop()
    jpeg = await loop.run_in_executor(None, scanner.stitcher.generate_preview)
    if jpeg is None:
        return JSONResponse({"error": "no tiles yet"}, status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")

@app.post("/api/scan/resume")
async def resume_scan():
    """Resume a scan that was paused by the global focus-failure gate."""
    if scanner.is_paused:
        scanner.resume_event.set()
        return {"status": "ok"}
    return {"status": "not_paused"}

@app.post("/api/scan/reimage/confirm")
async def confirm_reimage_selection(payload: ReimageSelectionPayload):
    """
    Submit which end-of-scan-audit-flagged tiles should actually be
    reimaged. Call after a 'reimage_review_required' WS event; unblocks
    the scan's reimage pass.
    """
    if not scanner.awaiting_reimage_review:
        return {"status": "not_awaiting"}

    valid = set(scanner.reimage_candidate_filenames)
    if payload.select_all:
        scanner.reimage_selected_filenames = list(scanner.reimage_candidate_filenames)
    else:
        scanner.reimage_selected_filenames = [f for f in payload.filenames if f in valid]

    scanner.reimage_decision_event.set()
    return {"status": "ok", "selected_count": len(scanner.reimage_selected_filenames)}

# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------
@app.get("/api/calibration/af_status")
async def af_status():
    """Return current autofocus calibration state for the UI."""
    if not autofocus:
        return {"available": False}
    return {
        "available":    True,
        "model_ready":  autofocus.is_ready,
        "calibrated":   autofocus.is_calibrated,
        "direction_sign": autofocus.config.direction_sign,
        "lut_bins_populated": sum(1 for b in autofocus.lut["bins"] if b["n_samples"] > 0),
        "calibration_date": autofocus.lut.get("calibration_date"),
    }

@app.post("/api/calibration/sweep")
async def run_calibration_sweep(bg: BackgroundTasks):
    """
    Run the focus sweep: sweeps the Canon ring ±sweep_range steps,
    captures on/off-axis frames at each position, builds the LUT,
    and detects the direction sign.

    Progress events stream via /ws/events:
        calibration_sweep_progress  — per step
        calibration_sweep_result    — on completion
    """
    if not autofocus:
        return {"status": "error", "message": "AutoFocus not available"}
    if scanner.is_scanning:
        return {"status": "error", "message": "Cannot calibrate while scan is running"}
    if not (autofocus.camera and autofocus.canon and autofocus.led):
        return {"status": "error", "message": "Hardware not connected (run Connect HW first)"}

    async def _run_sweep():
        loop = asyncio.get_running_loop()
        await autofocus.focus_sweep(loop, event_queue=scanner.event_queue)

    bg.add_task(_run_sweep)
    return {"status": "started"}

# ------------------------------------------------------------------
# Flat-field correction calibration
# ------------------------------------------------------------------
@app.get("/api/calibration/ffc/status")
async def ffc_status():
    """Return FFC calibration state for the UI on startup."""
    return {
        "loaded":      _ffc_loaded,
        "flatness_pct": round(ffc.flatness_pct(), 2) if _ffc_loaded else None,
    }

@app.post("/api/calibration/ffc/start")
async def ffc_start():
    """Begin FFC calibration flow — tells the frontend to prompt for dark frames."""
    await scanner.event_queue.put({"event": "ffc_dark_requested"})
    return {"status": "ok"}

@app.post("/api/calibration/ffc/capture_dark")
async def ffc_capture_dark():
    """
    Capture 4 dark SNAPs (lens capped / LED off) and store them.
    Emits ffc_flat_requested when done so the frontend advances to step 2.
    """
    global _ffc_dark_frames
    frames = []
    for _ in range(4):
        snap = await camera.snap(timeout=15.0)
        if snap:
            img = FlatFieldCorrector._decode_snap(snap)
            if img is not None:
                frames.append(img)
    if len(frames) < 2:
        return {"status": "error", "message": f"Only {len(frames)} dark frames captured — check camera connection"}
    _ffc_dark_frames = frames
    await scanner.event_queue.put({"event": "ffc_flat_requested"})
    return {"status": "ok", "frames_captured": len(frames)}

@app.post("/api/calibration/ffc/capture_flat")
async def ffc_capture_flat():
    """
    Capture 4 flat SNAPs from the user-selected clean-glass region.
    Micro-jogs between frames kill hot pixels; dust avoidance is the user's job
    (they already jogged to a clean area via the live stream before clicking).
    Emits ffc_complete with flatness_pct when done.
    """
    global _ffc_dark_frames, _ffc_loaded
    if not _ffc_dark_frames:
        return {"status": "error", "message": "No dark frames found — run dark capture first"}

    loop = asyncio.get_running_loop()
    frames = []
    # Small X jogs between frames: kill hot pixels, not intended to dodge dust
    jog_offsets = [0.0, 0.05, -0.05, 0.0]   # net displacement = 0 (returns to start)

    for dx in jog_offsets:
        if dx != 0.0:
            await loop.run_in_executor(None, cnc.jog, 'X', dx, 200)
            await loop.run_in_executor(None, cnc.wait_for_idle_blocking, 5.0)
        snap = await camera.snap(timeout=15.0)
        if snap:
            img = FlatFieldCorrector._decode_snap(snap)
            if img is not None:
                frames.append(img)

    if len(frames) < 2:
        return {"status": "error", "message": f"Only {len(frames)} flat frames captured — check camera connection"}

    ffc.set_dark(_ffc_dark_frames)
    ffc.set_flat(frames)
    ffc.save()

    # Inject into scanner so the running session uses the new calibration immediately
    scanner.ffc = ffc
    _ffc_loaded = True

    flatness = ffc.flatness_pct()
    await scanner.event_queue.put({
        "event":        "ffc_complete",
        "flatness_pct": round(flatness, 2) if flatness is not None else None,
    })
    return {"status": "ok", "flatness_pct": flatness}

# ------------------------------------------------------------------
# Live video WebSocket
# ------------------------------------------------------------------
@app.websocket("/ws/video")
async def video_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if camera.connected and camera.latest_frame:
                await ws.send_bytes(camera.latest_frame)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)