"""
CNC Controller (GRBL)
---------------------
Controls the CNC 3018 via GRBL firmware over serial.

Key changes from prescan version:
  - send_and_wait_ok()       : sends a G-code and blocks until GRBL acknowledges with 'ok'
  - wait_for_idle_blocking() : polls GRBL '?' status until state is 'Idle' (motion complete)
  - set_origin()             : G90 absolute mode + G92 X0 Y0 (set current pos as scan origin)
  - goto_xy()                : absolute G0 move to X, Y
  - All blocking methods have async equivalents via run_in_executor for asyncio scanner tasks.

The original fire-and-forget send_gcode() is preserved for prescan compatibility.
"""

import serial
import time
import asyncio


class CNCController:
    def __init__(self, port: str, baud_rate: int = 115200):
        self.port      = port
        self.baud_rate = baud_rate
        self.ser       = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2.5)            # Wait for GRBL to reset and emit welcome text
            self.ser.flushInput()      # Discard GRBL banner ("Grbl 1.1h ['$' for help]")
            self.send_and_wait_ok("G91")  # Relative mode for manual jogging / prescan
            print(f"[CNC] Connected on {self.port}")
            return True
        except Exception as e:
            print(f"[CNC] Connection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Low-level send (fire and forget — used by prescan)
    # ------------------------------------------------------------------
    def send_gcode(self, gcode: str) -> bool:
        if self.ser and self.ser.is_open:
            self.ser.write(f"{gcode}\n".encode('utf-8'))
            return True
        return False

    # ------------------------------------------------------------------
    # Send + block until GRBL 'ok' (command queued by planner)
    # ------------------------------------------------------------------
    def send_and_wait_ok(self, gcode: str, timeout: float = 5.0) -> bool:
        """
        Send a G-code command and read responses until GRBL sends 'ok'.
        Note: 'ok' means the command entered the motion planner, NOT that
        motion is complete. Call wait_for_idle_blocking() to confirm motion done.
        """
        if not self.ser or not self.ser.is_open:
            return False
        self.ser.write(f"{gcode}\n".encode('utf-8'))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
                if line == 'ok':
                    return True
                if line.startswith('error'):
                    print(f"[CNC] GRBL error '{line}' on command: {gcode}")
                    return False
                # Skip status reports <...> or other messages and keep reading
        print(f"[CNC] send_and_wait_ok timed out for: {gcode}")
        return False

    # ------------------------------------------------------------------
    # Poll GRBL '?' until state is Idle  (motion fully complete)
    # ------------------------------------------------------------------
    def wait_for_idle_blocking(self, timeout: float = 30.0) -> bool:
        """
        Repeatedly send '?' status query and parse GRBL response.
        Returns True when state reaches Idle, False on timeout.

        GRBL status format: <Idle|MPos:X,Y,Z|...>  or  <Run|MPos:X,Y,Z|...>
        """
        if not self.ser or not self.ser.is_open:
            return True   # No connection → nothing to wait for

        deadline = time.time() + timeout
        while time.time() < deadline:
            self.ser.write(b'?')
            time.sleep(0.06)          # Give GRBL 60 ms to queue the response
            raw = b''
            while self.ser.in_waiting:
                raw += self.ser.read(self.ser.in_waiting)
                time.sleep(0.01)      # Drain any trailing bytes
            text = raw.decode('utf-8', errors='replace')
            if 'Idle' in text:
                return True
            time.sleep(0.10)          # Poll at ~7 Hz

        print(f"[CNC] wait_for_idle timed out after {timeout:.0f}s")
        return False

    # ------------------------------------------------------------------
    # Scan-specific helpers (synchronous)
    # ------------------------------------------------------------------
    def set_origin(self):
        """
        Switch to absolute mode and zero the work coordinate at the current position.
        Call this once at scan start after the user has manually positioned the stage.
        All subsequent goto_xy() calls are relative to this point.
        """
        self.send_and_wait_ok("G90")         # Absolute positioning mode
        self.send_and_wait_ok("G92 X0 Y0")   # Current position = (0, 0) in work coords
        print("[CNC] Origin set. Absolute positioning active.")

    def goto_xy(self, x_mm: float, y_mm: float, feedrate: int = 400):
        """
        Move to absolute (x_mm, y_mm). Requires G90 mode (set by set_origin).
        Does NOT wait for motion to complete — follow with wait_for_idle_blocking().
        """
        self.send_and_wait_ok(f"G0 X{x_mm:.4f} Y{y_mm:.4f} F{feedrate}")

    def return_to_origin(self, feedrate: int = 600):
        """Move back to (0, 0) and restore relative mode for jogging / next prescan."""
        self.send_and_wait_ok(f"G0 X0 Y0 F{feedrate}")
        # wait_for_idle_blocking() is expected to be called after this

    def restore_relative_mode(self):
        """Switch back to G91 relative mode after scan completes."""
        self.send_and_wait_ok("G91")
        print("[CNC] Relative positioning mode restored.")

    def manual_jog(self, axis: str, distance: float, feedrate: int = 500) -> bool:
        """
        Execute an incremental manual jog safely during a scan.

        The scan path keeps GRBL in G90 absolute mode. Manual jogging is easier
        for the operator in G91 incremental mode, so switch to G91 only for the
        jog and restore G90 immediately afterward.
        """
        if not self.ser or not self.ser.is_open:
            print("[CNC] Manual jog requested but serial port is not open.")
            return False

        if not self.wait_for_idle_blocking():
            print("[CNC] Manual jog aborted: controller did not reach Idle.")
            return False

        if not self.send_and_wait_ok("G91"):
            print("[CNC] Manual jog aborted: failed to enter G91 mode.")
            return False

        try:
            if not self.send_and_wait_ok(f"G0 {axis}{distance} F{feedrate}"):
                print(f"[CNC] Manual jog failed: axis={axis} distance={distance}.")
                return False

            if not self.wait_for_idle_blocking():
                print("[CNC] Manual jog failed: motion did not complete.")
                return False

            return True
        finally:
            if not self.send_and_wait_ok("G90"):
                print("[CNC] Warning: failed to restore G90 mode after manual jog.")

    # ------------------------------------------------------------------
    # Prescan-compatible jog (unchanged)
    # ------------------------------------------------------------------
    def jog(self, axis: str, distance: float, feedrate: int = 500):
        """Relative jog. e.g. jog('X', 1.5)  —  requires G91 mode."""
        if self.ser and self.ser.is_open:
            self.send_gcode(f"G0 {axis}{distance} F{feedrate}")

    # ------------------------------------------------------------------
    # Async wrappers  (use from asyncio scan loop via run_in_executor)
    # ------------------------------------------------------------------
    async def async_goto_xy(self, x_mm: float, y_mm: float, feedrate: int = 400):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.goto_xy, x_mm, y_mm, feedrate)

    async def async_wait_for_idle(self, timeout: float = 30.0):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.wait_for_idle_blocking, timeout)

    async def async_set_origin(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.set_origin)

    async def async_return_to_origin(self, feedrate: int = 600):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.return_to_origin, feedrate)
