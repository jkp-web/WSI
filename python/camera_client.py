"""
Camera Client
-------------
TCP client that connects to server.py running on the Raspberry Pi Zero 2W.

Binary packet protocol (5-byte header):
    1 byte  : message type  (1 = low-res stream frame | 2 = high-res SNAP)
    4 bytes : payload length (big-endian uint32)

New in Phase 1:
  - snap()            : request a full-resolution JPEG from the Pi and return the bytes
  - snap_event / snap_data : asyncio-safe signalling for the receive loop

Usage:
    camera = CameraClient('192.168.0.4', 8000)
    await camera.connect()
    ...
    jpeg_bytes = await camera.snap(timeout=15.0)   # during scan tile capture
"""

import asyncio
import struct


class CameraClient:
    def __init__(self, host: str = '192.168.0.4', port: int = 8000):
        self.host   = host
        self.port   = port
        self.reader = None
        self.writer = None
        self.connected   = False
        self.latest_frame = b''      # Updated every low-res stream packet (msg_type 1)

        self._receive_task = None
        self.new_frame_event = asyncio.Event()

        # High-res SNAP signalling  (set by _receive_loop when msg_type 2 arrives)
        self.snap_event = asyncio.Event()
        self.snap_data  = b''

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    async def connect(self) -> bool:
        try:
            print(f"[Camera] Connecting to {self.host}:{self.port}...")
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self.connected = True
            print("[Camera] Connected successfully.")
            self._receive_task = asyncio.create_task(self._receive_loop())
            return True
        except Exception as e:
            print(f"[Camera] Connection failed: {e}")
            return False

    async def disconnect(self):
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        print("[Camera] Disconnected.")

    # ------------------------------------------------------------------
    # Receive loop  (background task, handles both frame types)
    # ------------------------------------------------------------------
    async def _receive_loop(self):
        """Continuously pull packets from the Pi TCP stream."""
        while self.connected:
            try:
                # Fixed 5-byte header: 1B type + 4B length
                header = await self.reader.readexactly(5)
                msg_type, msg_len = struct.unpack('>BL', header)
                data = await self.reader.readexactly(msg_len)

                if msg_type == 1:
                    # Low-res stream frame (640×480 JPEG)
                    self.latest_frame = data
                    self.new_frame_event.set()
                    self.new_frame_event.clear()

                elif msg_type == 2:
                    # High-res SNAP response — signal snap() that data has arrived
                    self.snap_data = data
                    self.snap_event.set()

            except asyncio.IncompleteReadError:
                print("[Camera] Stream disconnected.")
                self.connected = False
                break
            except Exception as e:
                print(f"[Camera] Receive loop error: {e}")
                self.connected = False
                break

    # ------------------------------------------------------------------
    # High-res snapshot (Phase 1 scan capture)
    # ------------------------------------------------------------------
    async def snap(self, timeout: float = 15.0) -> bytes | None:
        """
        Request a full-resolution JPEG from the Pi and wait for the response.

        The Pi captures from the main (4056×3040) stream, encodes at JPEG q80,
        and sends it back as a msg_type=2 packet. This typically takes 2–5 s on
        a Pi Zero 2W — use the timeout accordingly.

        Returns:
            JPEG bytes on success, None on timeout or if not connected.
        """
        if not self.connected or not self.writer:
            print("[Camera] snap() called but camera not connected.")
            return None

        # Clear any stale SNAP signal before requesting
        self.snap_event.clear()
        self.snap_data = b''

        # Send SNAP command to Pi
        self.writer.write(b"SNAP\n")
        await self.writer.drain()

        try:
            await asyncio.wait_for(self.snap_event.wait(), timeout=timeout)
            return self.snap_data
        except asyncio.TimeoutError:
            print(f"[Camera] SNAP response timed out after {timeout:.0f}s")
            return None
