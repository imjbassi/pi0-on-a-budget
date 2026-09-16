"""
fake_serial.py — a stand-in for serial.Serial that emits teleop.ino's stream.

Lets the whole recording pipeline run with no Arduino attached: a header line,
then "<millis>,<base>,<shoulder>,<elbow>,<gripper>" at 50 Hz, with smooth
motion inside teleop.ino's JOINT_MIN/JOINT_MAX limits.

This is only for exercising the software. Its timing is far cleaner than a
real Nano over a USB-serial adapter, so it says nothing about real jitter.
"""

import math
import time

JOINT_MIN = [0, 15, 0, 20]
JOINT_MAX = [180, 165, 180, 160]


class FakeSerial:
    def __init__(self, rate_hz=50.0, port="FAKE", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.period = 1.0 / rate_hz
        self.start = time.perf_counter()
        self.tick = 0
        self.sent_header = False

    def _angles(self, t):
        raw = [
            90 + 70 * math.sin(2 * math.pi * 0.20 * t),
            90 + 50 * math.sin(2 * math.pi * 0.13 * t + 1.0),
            90 + 60 * math.sin(2 * math.pi * 0.17 * t + 2.0),
            160 if math.sin(2 * math.pi * 0.10 * t) > 0 else 20,
        ]
        return [int(min(max(a, lo), hi)) for a, lo, hi in zip(raw, JOINT_MIN, JOINT_MAX)]

    def readline(self):
        if not self.sent_header:
            self.sent_header = True
            return b"millis,base,shoulder,elbow,gripper\r\n"

        due = self.start + self.tick * self.period
        now = time.perf_counter()
        if now < due:
            time.sleep(due - now)

        t = self.tick * self.period
        self.tick += 1
        millis = int(round(t * 1000))
        fields = [str(millis)] + [str(a) for a in self._angles(t)]
        return (",".join(fields) + "\r\n").encode()

    def reset_input_buffer(self):
        pass

    def close(self):
        pass
