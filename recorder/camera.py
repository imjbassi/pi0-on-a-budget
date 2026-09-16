"""
camera.py — webcam capture with host timestamps, for episode recording.

The camera runs continuously from startup (so exposure has settled and there
is no open-device delay when an episode starts), but frames are only written
to disk while an episode is recording.

Timing: each frame is stamped the moment read() returns, on the same clocks
record_episode.py uses for serial rows (time.perf_counter() and time.time()).
That stamp includes camera exposure + USB transfer + decode latency — a roughly
constant offset that is NOT corrected here. It has to be measured against the
physical arm (see README), and applied at conversion time.

Frames are JPEG-encoded on a separate writer thread so encoding never stalls
capture. If the writer falls behind, frames are dropped and counted rather
than silently lost.
"""

import csv
import os
import queue
import threading
import time

import cv2
import numpy as np


BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class OpenCVSource:
    """A real camera opened through OpenCV."""

    def __init__(self, index=0, width=1280, height=720, fps=30, backend="dshow"):
        self.index = index
        self.backend = backend
        self.cap = cv2.VideoCapture(index, BACKENDS[backend])
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index} with backend '{backend}'. "
                "Run with --list-cameras, and for the Sony ZV-1 make sure USB "
                "Streaming mode is enabled on the camera."
            )
        # Requests only — the driver may silently pick something else.
        # actual_settings() reports what was really negotiated.
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, image = self.cap.read()
        return image if ok else None

    def actual_settings(self):
        return {
            "source": f"opencv:{self.index}:{self.backend}",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported_by_driver": float(self.cap.get(cv2.CAP_PROP_FPS)),
        }

    def close(self):
        self.cap.release()


class FakeSource:
    """Synthetic frames at a fixed rate, for testing without a camera."""

    def __init__(self, width=640, height=360, fps=30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.period = 1.0 / fps
        self.next_due = time.perf_counter()
        self.count = 0

    def read(self):
        now = time.perf_counter()
        if now < self.next_due:
            time.sleep(self.next_due - now)
        # Schedule off the ideal grid, but don't try to "catch up" with a burst
        # if we fell far behind — a real camera wouldn't either.
        self.next_due = max(self.next_due + self.period, time.perf_counter() - self.period)

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = ((self.count * 7) % 256, 80, 160)
        cv2.putText(image, str(self.count), (10, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.count += 1
        return image

    def actual_settings(self):
        return {
            "source": "fake",
            "width": self.width,
            "height": self.height,
            "fps_reported_by_driver": self.fps,
        }

    def close(self):
        pass


def list_cameras(max_index=6, backend="dshow"):
    """Probe camera indices and report what opens and delivers a frame."""
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, BACKENDS[backend])
        if not cap.isOpened():
            cap.release()
            continue
        ok, image = cap.read()
        found.append({
            "index": index,
            "delivers_frames": bool(ok),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported_by_driver": float(cap.get(cv2.CAP_PROP_FPS)),
        })
        cap.release()
    return found


class CameraRecorder:
    FRAMES_CSV_FIELDS = ["frame_index", "host_time", "host_mono", "file"]

    def __init__(self, source, jpeg_quality=90, max_queue=120):
        self.source = source
        self.jpeg_quality = jpeg_quality

        self.queue = queue.Queue(maxsize=max_queue)
        self.lock = threading.Lock()
        self.running = False
        self.recording = False

        self.frames_seen = 0          # all frames read, recording or not
        self.read_failures = 0
        self.last_frame_mono = None
        self.latest = None            # (image, host_mono) of the newest frame

        self._reset_episode(None)

    def _reset_episode(self, frames_dir):
        self.frames_dir = frames_dir
        self.next_index = 0
        self.saved_rows = []
        self.dropped = 0
        self.write_failures = 0

    # ---------------------------------------------------------- lifecycle

    def start(self, first_frame_timeout=10.0):
        """Start capture + writer threads and wait for the first frame."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.capture_thread.start()
        self.writer_thread.start()

        deadline = time.perf_counter() + first_frame_timeout
        while self.last_frame_mono is None:
            if time.perf_counter() > deadline:
                self.stop()
                raise RuntimeError(
                    f"No frames from camera after {first_frame_timeout:.0f}s "
                    f"({self.read_failures} failed reads)."
                )
            time.sleep(0.01)

    def stop(self):
        self.running = False
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=2)
        self.queue.put(None)          # writer shutdown sentinel
        if hasattr(self, "writer_thread"):
            self.writer_thread.join(timeout=5)
        self.source.close()

    def begin_episode(self, frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        with self.lock:
            self._reset_episode(frames_dir)
            self.recording = True

    def end_episode(self):
        """Stop saving, wait for pending writes, and return per-episode stats."""
        with self.lock:
            self.recording = False
        self.queue.join()
        return {
            "frames_saved": len(self.saved_rows),
            "frames_dropped_writer_behind": self.dropped,
            "write_failures": self.write_failures,
            "rows": list(self.saved_rows),
        }

    # ------------------------------------------------------------ threads

    def _capture_loop(self):
        while self.running:
            image = self.source.read()
            host_mono = time.perf_counter()
            host_time = time.time()

            if image is None:
                self.read_failures += 1
                time.sleep(0.005)
                continue

            self.frames_seen += 1
            self.last_frame_mono = host_mono
            self.latest = (image, host_mono)          # single assignment: safe to read from other threads

            with self.lock:
                if not self.recording:
                    continue
                index = self.next_index
                try:
                    self.queue.put_nowait((self.frames_dir, index, image, host_time, host_mono))
                    self.next_index += 1
                except queue.Full:
                    self.dropped += 1

    def _writer_loop(self):
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            frames_dir, index, image, host_time, host_mono = item
            name = f"frame_{index:06d}.jpg"
            if cv2.imwrite(os.path.join(frames_dir, name), image, params):
                self.saved_rows.append({
                    "frame_index": index,
                    "host_time": host_time,
                    "host_mono": host_mono,
                    "file": name,
                })
            else:
                self.write_failures += 1
            self.queue.task_done()


def write_frames_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CameraRecorder.FRAMES_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
