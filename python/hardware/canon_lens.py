import serial
import time
import sys


class CanonLens:
    def __init__(self, port, baud_rate=9600):
        """
        Arduino-controlled Canon lens focus ring.

        Phase 2 addition: `self.position` tracks cumulative ring steps from
        the power-on zero point.  AutoFocus uses this to detect when the ring
        is approaching its travel limit and trigger a CNC-Z recentre.

        Args:
            port (str): Serial port of the lens Arduino  (e.g. '/dev/ttyUSB0').
            baud_rate (int): Must match the Arduino sketch  (default 9600).
        """
        self.port      = port
        self.baud_rate = baud_rate
        self.ser       = None
        self.position  = 0      # cumulative ring steps (+ = near, − = far)

        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"[CanonLens] Connected to {self.port}.")
        except serial.SerialException as e:
            print(f"[CanonLens] Error connecting to {self.port}: {e}")
            sys.exit(1)

    def _send_command(self, command_str):
        if self.ser and self.ser.is_open:
            self.ser.write((command_str + '\n').encode('utf-8'))
            time.sleep(0.1)
            return self.read_response()
        print("[CanonLens] Serial port not open.")
        return None

    def read_response(self):
        responses = []
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                responses.append(line)
        return responses

    def focus(self, steps: int):
        """
        Move the focus ring by `steps`.
        Positive → near (minimum focus).
        Negative → far (infinity).
        Updates self.position so AutoFocus can track ring travel.
        """
        if steps == 0:
            return
        cmd = f"+{steps}" if steps > 0 else f"{steps}"
        result = self._send_command(cmd)
        self.position += steps     # track cumulative travel
        return result

    def reset_position(self):
        """Call after a CNC-Z recentre to zero the tracked position."""
        self.position = 0

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[CanonLens] Connection closed.")
