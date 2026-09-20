"""
openpi_configs.py — gello-lite policy transforms and training configs for openpi.

Modeled on openpi's libero_policy.py and LeRobotLiberoDataConfig, with these
deliberate differences (each one a departure from openpi's documented recipe):

  1. One camera. The Logitech Brio 101 frame is base_0_rgb; both wrist slots are zero images.
  2. 4-dim state/actions: base, shoulder, elbow (rad) + gripper (0..1).
  3. Absolute joint targets in the dataset -> DeltaActions on the 3 joints,
     gripper stays absolute (openpi's convention).
  4. LoRA freeze filter ALSO freezes the SigLIP image encoder. openpi's stock
     π0-FAST LoRA filter trains it in full (~400M params with AdamW state),
     which cannot fit in 12 GB. Frozen params are cast to bfloat16 by openpi's
     train.py.
  5. Fresh normalization stats (no pretraining embodiment resembles this arm).

Registered by gello_pi0.run before openpi's own scripts parse their CLI.
"""

import dataclasses
import os
import pathlib

import flax.nnx as nnx
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_fast
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import weight_loaders

RUNS_DIR = pathlib.Path(os.environ.get("GELLO_RUNS_DIR", "/mnt/d/pi0-on-a-budget-runs"))
ACTION_DIM = 4
PI0_FAST_BASE = "gs://openpi-assets/checkpoints/pi0_fast_base/params"


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class GelloInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        image = _parse_image(data["observation/image"])
        # pi0-FAST: padding images are NOT masked out (see libero_policy.py); pi0: they are.
        pad_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": pad_mask,
                "right_wrist_0_rgb": pad_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class GelloOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LeRobotGelloDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(
            inputs=[GelloInputs(model_type=model_config.model_type)],
            outputs=[GelloOutputs()],
        )
        delta_mask = _transforms.make_bool_mask(3, -1)      # joints delta, gripper absolute
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def make_model_config(action_horizon=15):
    return pi0_fast.Pi0FASTConfig(
        action_dim=ACTION_DIM,
        action_horizon=action_horizon,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    )


def make_freeze_filter(model_config, freeze_vision=True):
    base = model_config.get_freeze_filter()           # LLM weights except LoRA
    if not freeze_vision:
        return base
    return nnx.Any(base, nnx_utils.PathRegex(".*img.*"))


def make_train_config(name, repo_id, *, action_horizon=15, freeze_vision=True, batch_size=1,
                      num_train_steps=3000, save_interval=500):
    model = make_model_config(action_horizon)
    return _config.TrainConfig(
        name=name,
        model=model,
        data=LeRobotGelloDataConfig(
            repo_id=repo_id,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI0_FAST_BASE),
        freeze_filter=make_freeze_filter(model, freeze_vision),
        ema_decay=None,                                # openpi: EMA off for LoRA
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        keep_period=None,
        log_interval=25,
        num_workers=2,
        wandb_enabled=False,
        assets_base_dir=str(RUNS_DIR / "assets"),
        checkpoint_base_dir=str(RUNS_DIR / "checkpoints"),
    )


CONFIGS = [
    # Synthetic data: pipeline + 12 GB memory test.
    make_train_config("gello_fake_lora", "local/gello_fake"),
    # Stock openpi freeze filter (image encoder trainable), for documenting the memory difference.
    make_train_config("gello_fake_lora_vision_trainable", "local/gello_fake", freeze_vision=False),
    # Real gello-lite data.
    make_train_config("gello_lora", "local/gello"),
]


def register():
    for cfg in CONFIGS:
        _config._CONFIGS_DICT[cfg.name] = cfg
