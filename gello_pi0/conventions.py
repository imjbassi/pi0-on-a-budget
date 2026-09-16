"""
conventions.py — the single definition of how gello-lite numbers map to π0 numbers.

Used by the dataset converter (WSL), open-loop eval (WSL) and the closed-loop
controller (Windows). Keeping it in one place is what guarantees the model sees
the same thing in training and on the real arm.

Joint mapping (openpi docs/norm_stats.md conventions):
  - base, shoulder, elbow: degrees -> radians, relative to a per-joint zero angle
  - gripper: degrees -> [0, 1], 0.0 = fully open, 1.0 = fully closed

Image mapping: BGR (OpenCV) or RGB frame -> RGB uint8, letterboxed to 224x224
with Pillow bilinear resize — the same resize-with-pad openpi applies at
inference, so storing 224x224 means openpi's own resize is a no-op.
"""

import dataclasses
import json

import numpy as np
from PIL import Image

JOINTS = ("base", "shoulder", "elbow", "gripper")
IMAGE_SIZE = 224


@dataclasses.dataclass(frozen=True)
class RobotConvention:
    # Servo angle (degrees) treated as 0 rad for base/shoulder/elbow.
    joint_zero_deg: tuple = (90.0, 90.0, 90.0)
    # Gripper servo angles. MUST be confirmed on the real arm — see config/robot.json.
    gripper_open_deg: float | None = None
    gripper_closed_deg: float | None = None
    # Hard limits, copied from teleop.ino JOINT_MIN / JOINT_MAX.
    joint_min_deg: tuple = (0, 15, 0, 20)
    joint_max_deg: tuple = (180, 165, 180, 160)
    confirmed_on_hardware: bool = False

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = json.load(f)
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        for key in ("joint_zero_deg", "joint_min_deg", "joint_max_deg"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def require_gripper(self):
        if self.gripper_open_deg is None or self.gripper_closed_deg is None:
            raise ValueError(
                "Gripper open/closed angles are not set in the robot config. "
                "Confirm them on the assembled arm before converting real data."
            )
        if self.gripper_open_deg == self.gripper_closed_deg:
            raise ValueError("gripper_open_deg and gripper_closed_deg must differ")

    # ---------------------------------------------------------- joints

    def deg_to_model(self, deg):
        """(..., 4) servo degrees -> (..., 4) float32 [rad, rad, rad, gripper 0..1]."""
        self.require_gripper()
        deg = np.asarray(deg, dtype=np.float64)
        out = np.empty(deg.shape, dtype=np.float64)
        out[..., :3] = np.deg2rad(deg[..., :3] - np.asarray(self.joint_zero_deg))
        span = self.gripper_closed_deg - self.gripper_open_deg
        out[..., 3] = np.clip((deg[..., 3] - self.gripper_open_deg) / span, 0.0, 1.0)
        return out.astype(np.float32)

    def model_to_deg(self, values, clamp=True):
        """(..., 4) model units -> (..., 4) servo degrees (float). Inverse of deg_to_model."""
        self.require_gripper()
        values = np.asarray(values, dtype=np.float64)
        out = np.empty(values.shape, dtype=np.float64)
        out[..., :3] = np.rad2deg(values[..., :3]) + np.asarray(self.joint_zero_deg)
        grip = np.clip(values[..., 3], 0.0, 1.0)
        out[..., 3] = self.gripper_open_deg + grip * (self.gripper_closed_deg - self.gripper_open_deg)
        if clamp:
            out = np.clip(out, np.asarray(self.joint_min_deg), np.asarray(self.joint_max_deg))
        return out


# ---------------------------------------------------------------- images

def preprocess_image(image, bgr=True, size=IMAGE_SIZE):
    """HxWx3 uint8 frame -> size x size x 3 RGB uint8, aspect preserved with black padding."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    if bgr:
        image = image[..., ::-1]
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = Image.fromarray(np.ascontiguousarray(image)).resize((new_w, new_h), Image.BILINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(resized)
    return canvas


def degrees_error(a_deg, b_deg):
    """Per-joint absolute error in degrees; gripper included as servo degrees."""
    return np.abs(np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64))
