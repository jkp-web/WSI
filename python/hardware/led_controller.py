"""
LED Matrix Controller
---------------------
Controls the 8x8 LED illumination matrix via a dedicated Arduino.
Serial protocol: send "1\n" for on-axis brightfield, "2\n" for off-axis AF illumination.

Port is separate from the Canon lens Arduino (/dev/ttyUSB0).
Typical port: /dev/ttyUSB2 or /dev/ttyACM0 — confirm with `ls /dev/ttyUSB*`

Illumination modes:
    MODE_ON_AXIS  (1) — Full matrix on, even illumination for brightfield image capture.
    MODE_OFF_AXIS (2) — Single/few edge LEDs at angle, for FCFNN autofocus frame capture.
"""

import serial
import time
import asyncio
from arduino.app_utils import Bridge


class LedController:
    MODE_ON_AXIS  = 1   # Brightfield: full uniform illumination
    MODE_OFF_AXIS = 2   # Autofocus:   angled off-axis illumination (required by FCFNN)

    def __init__(self, port: str, baud_rate: int = 9600, settle_time: float = 0.25):
        """
        Args:
            port:        Serial port of the LED Arduino  (e.g. '/dev/ttyUSB2')
            baud_rate:   Must match the Arduino sketch    (default 9600)
            settle_time: Seconds to wait after a mode switch for LED/camera to stabilise.
                         The FCFNN is illumination-sensitive; 250ms is a safe floor.
        """
        self.port        = port
        self.baud_rate   = baud_rate
        self.settle_time = settle_time
        self.ser         = None
        self.current_mode: int | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            
            return True
        except Exception as e:
            print(f"[LED] Connection failed on {self.port}: {e}")
            return False

    def close(self):
        
            print("[LED] Disconnected.")

    # ------------------------------------------------------------------
    # Synchronous mode switching
    # ------------------------------------------------------------------
    def set_mode(self, mode: int) -> bool:
        """
        Send illumination mode command and block for settle_time.
        Returns True on success.
        """
        
        try:
            print(mode)
            print(type(mode))
            mod = 1
            print(type(mod))
            Bridge.call("centers",mode)
            time.sleep(self.settle_time)    # Camera exposure must stabilise after switch
            self.current_mode = mode
            label = "ON-AXIS" if mode == self.MODE_ON_AXIS else "OFF-AXIS"
            print(f"[LED] Mode set → {label}")
            return True
        except Exception as e:
            print(f"[LED] set_mode error: {e}")
            return False

    def on_axis(self)  -> bool: return self.set_mode(self.MODE_ON_AXIS)
    def off_axis(self) -> bool: return self.set_mode(self.MODE_OFF_AXIS)

    # ------------------------------------------------------------------
    # Async wrappers (use from asyncio scanner tasks via run_in_executor)
    # ------------------------------------------------------------------
    async def async_set_mode(self, mode: int):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.set_mode, mode)

    async def async_on_axis(self):  await self.async_set_mode(self.MODE_ON_AXIS)
    async def async_off_axis(self): await self.async_set_mode(self.MODE_OFF_AXIS)
