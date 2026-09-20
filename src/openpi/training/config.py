"""Training settings used by the MTO phase trainer."""

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.models import model as _model
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    model: _model.BaseModelConfig
    exp_name: str
    project_name: str = "mto"
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: nnx.filterlib.Filter = dataclasses.field(default_factory=nnx.Nothing)
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    log_gradient_diagnostics: bool = False
    fsdp_devices: int = 1

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
