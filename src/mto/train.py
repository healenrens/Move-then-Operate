from __future__ import annotations

import dataclasses
import functools
import logging
import json
from pathlib import Path
import random
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Iterator, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tyro

from flax.training import common_utils

from openpi.models import model as _model
from openpi.models.mto_config import DualExpertPi0Config, TrainingMode, create_mto_config
from openpi.models.tokenizer import PaligemmaTokenizer
from mto.train_utils import init_logging, init_wandb, init_train_state, train_step as base_train_step
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders
import openpi.training.utils as training_utils

from mto.dataset import PhaseDataset, wide_camera_collate_fn


@dataclasses.dataclass
class WideCameraTrainConfig:
    data_root: str
    exp_name: str
    project_name: str = "openpi"
    checkpoint_base_dir: str = "./checkpoints"

    batch_size: int = 8
    num_steps: int = 10000
    lr: float = 5e-5
    lr_final: float = 1e-5
    lr_warmup: int = 1000
    weight_decay: float = 1e-8
    ema_decay: float | None = 0.99

    action_horizon: int = 30
    model_action_dim: int = 32
    max_token_len: int = 50

    training_mode: TrainingMode = "full"
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None

    num_workers: int = 8
    seed: int = 42
    wandb_enabled: bool = False

    save_interval: int = 1000
    log_interval: int = 10
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = True
    fsdp_devices: int = 1

    init_params: str | None = None
    init_source: Literal["pi0_base", "dual_expert"] = "pi0_base"
    tokenizer_path: str | None = None

    data_type: Literal["qpos"] = "qpos"
    data_format: Literal["auto", "hdf5", "lerobot_v3", "lerobot"] = "lerobot_v3"
    labels_root: str | None = None
    task_names: tuple[str, ...] = ()
    split_names: tuple[str, ...] = ()
    drop_small_action_deltas: bool = False
    min_action_magnitude: float = 1e-5
    normalize_method: Literal["zscore", "min_max"] = "zscore"
    long_phase_ratio: float = 0.5

    log_gradient_diagnostics: bool = False
    freeze_paligemma: bool = False
    move_norm_stats_path: str | None = None
    operate_norm_stats_path: str | None = None


def _create_train_config(cfg: WideCameraTrainConfig) -> _config.TrainConfig:
    assert cfg.batch_size % jax.device_count() == 0, "Global batch size must divide across devices"

    if cfg.init_params is not None:
        weight_loader: _weight_loaders.WeightLoader = _weight_loaders.MtoWeightLoader(cfg.init_params, cfg.init_source)
    else:
        weight_loader = _weight_loaders.NoOpWeightLoader()

    lr_schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=cfg.lr_warmup,
        peak_lr=cfg.lr,
        decay_steps=cfg.num_steps,
        decay_lr=cfg.lr_final,
    )

    model_config: DualExpertPi0Config = create_mto_config(
        cfg.training_mode,
        action_dim=cfg.model_action_dim,
        action_horizon=cfg.action_horizon,
        max_token_len=cfg.max_token_len,
        paligemma_variant=cfg.paligemma_variant,
        action_expert_variant=cfg.action_expert_variant,
    )
    if cfg.freeze_paligemma:
        model_config = dataclasses.replace(model_config, freeze_paligemma=True)

    return _config.TrainConfig(
        name="wide_camera_dual_expert",
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        model=model_config,
        freeze_filter=model_config.get_freeze_filter(),
        weight_loader=weight_loader,
        lr_schedule=lr_schedule,
        optimizer=_optimizer.AdamW(weight_decay=cfg.weight_decay),
        ema_decay=cfg.ema_decay,
        checkpoint_base_dir=cfg.checkpoint_base_dir,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_train_steps=cfg.num_steps,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        keep_period=cfg.keep_period,
        overwrite=cfg.overwrite,
        resume=cfg.resume,
        wandb_enabled=cfg.wandb_enabled,
        fsdp_devices=cfg.fsdp_devices,
        log_gradient_diagnostics=cfg.log_gradient_diagnostics,
    )


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _pad_to_dim(array: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = array.shape[-1]
    if current_dim == target_dim:
        return array
    if current_dim > target_dim:
        return array[..., :target_dim]
    pad = torch.zeros(*array.shape[:-1], target_dim - current_dim, dtype=array.dtype, device=array.device)
    return torch.cat([array, pad], dim=-1)


def _tokenize_instructions(tokenizer: PaligemmaTokenizer, instructions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tokens_list = []
    masks_list = []
    for instr in instructions:
        prompt = f"{(instr or '').strip().replace('_', ' ')}\n"
        tokens, mask = tokenizer.tokenize(prompt)
        tokens_list.append(tokens)
        masks_list.append(mask)
    tokens = np.stack(tokens_list, axis=0).astype(np.int32)
    masks = np.stack(masks_list, axis=0).astype(bool)
    return tokens, masks


def _convert_optional_mask(mask_tensor: torch.Tensor | None) -> np.ndarray | None:
    if mask_tensor is None:
        return None
    return _torch_to_numpy(mask_tensor).astype(bool)


def _pad_action_dim_mask(mask: np.ndarray | None, target_dim: int) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape[-1] >= target_dim:
        return mask[..., :target_dim]
    pad_shape = mask.shape[:-1] + (target_dim - mask.shape[-1],)
    return np.concatenate([mask, np.zeros(pad_shape, dtype=bool)], axis=-1)


def _build_observation_and_actions(
    batch: dict,
    tokenizer: PaligemmaTokenizer,
    target_action_dim: int,
    action_horizon: int,
) -> tuple[_model.Observation, np.ndarray]:
    obs = batch["observation"]

    obs_images = obs["image"]
    images: dict[str, np.ndarray] = {}
    for key_src, key_dst in {
        "head_camera": "base_0_rgb",
        "left_camera": "left_wrist_0_rgb",
        "right_camera": "right_wrist_0_rgb",
    }.items():
        if key_src not in obs_images:
            continue
        np_img = _torch_to_numpy(obs_images[key_src])
        np_img = np.transpose(np_img, (0, 2, 3, 1))
        images[key_dst] = np_img.astype(np.float32)

    if not images:
        raise ValueError("No supported camera keys found in batch observation images.")

    batch_size = batch["action"].shape[0]
    image_masks = {key: np.ones(batch_size, dtype=bool) for key in images}

    state = _pad_to_dim(obs["state"], target_action_dim)
    state_np = _torch_to_numpy(state).astype(np.float32)

    actions = _pad_to_dim(batch["action"], target_action_dim)
    actions_np = _torch_to_numpy(actions).astype(np.float32)

    instructions = obs["instr"]
    tokens, masks = _tokenize_instructions(tokenizer, instructions)
    token_loss_mask = np.zeros_like(masks, dtype=bool)

    route_labels_tensor = batch["route_label"]
    route_labels = _torch_to_numpy(route_labels_tensor).astype(np.int32)
    action_loss_mask = _convert_optional_mask(batch.get("action_loss_mask"))
    action_dim_mask = _pad_action_dim_mask(_convert_optional_mask(batch.get("action_dim_mask")), target_action_dim)

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state_np,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=masks,
        token_ar_mask=None,
        token_loss_mask=token_loss_mask,
        route_labels=route_labels,
        action_loss_mask=action_loss_mask,
        action_dim_mask=action_dim_mask,
    )

    return observation, actions_np.reshape(-1, action_horizon, target_action_dim)


def _shard_observation(
    observation: _model.Observation,
    sharding: jax.sharding.NamedSharding,
) -> _model.Observation:
    def _convert(x):
        if isinstance(x, np.ndarray):
            return jax.make_array_from_process_local_data(sharding, x)
        return x

    return jax.tree.map(_convert, observation)


def _shard_actions(actions: np.ndarray, sharding: jax.sharding.NamedSharding) -> jax.Array:
    return jax.make_array_from_process_local_data(sharding, actions)


def create_dataset(cfg: WideCameraTrainConfig) -> PhaseDataset:
    return PhaseDataset(
        data_dir=cfg.data_root, data_format=cfg.data_format, labels_root=cfg.labels_root,
        task_names=cfg.task_names, split_names=cfg.split_names, action_steps=cfg.action_horizon,
        normalize_method=cfg.normalize_method, long_phase_ratio=cfg.long_phase_ratio,
        drop_small_action_deltas=cfg.drop_small_action_deltas, min_action_magnitude=cfg.min_action_magnitude,
        move_norm_stats_path=cfg.move_norm_stats_path, operate_norm_stats_path=cfg.operate_norm_stats_path,
    )


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_data_iterator(
    cfg: WideCameraTrainConfig,
    tokenizer: PaligemmaTokenizer,
    sharding: jax.sharding.NamedSharding,
    dataset: PhaseDataset | None = None,
) -> Iterator[tuple[_model.Observation, jax.Array]]:
    if dataset is None:
        dataset = create_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg.seed + jax.process_index())
    local_batch_size = cfg.batch_size // jax.process_count()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        # Samples choose a phase independently of the index. Keep full batches
        # available even when a small dataset has fewer phases than the batch.
        sampler=torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=max(len(dataset), local_batch_size), generator=generator
        ),
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        multiprocessing_context="spawn" if cfg.num_workers else None,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=wide_camera_collate_fn,
        drop_last=True,
    )

    while True:
        for raw_batch in loader:
            obs_np, actions_np = _build_observation_and_actions(
                raw_batch,
                tokenizer,
                cfg.model_action_dim,
                cfg.action_horizon,
            )
            sharded_obs = _shard_observation(obs_np, sharding)
            sharded_actions = _shard_actions(actions_np, sharding)
            yield sharded_obs, sharded_actions


def record_run(cfg, train_config, dataset, *, resuming):
    root = train_config.checkpoint_dir
    payload = dataclasses.asdict(cfg)
    payload["normalization_mode"] = "per-expert"
    payload.update(raw_action_dim=14, gripper_indices=[6, 13],
                   action_representation="joint14_delta_joint_absolute_gripper",
                   phase_end_convention="inclusive_state_frame", sampling_clock="raw_episode_frames")
    payload["model"] = dataclasses.asdict(train_config.model)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=False)
    payload["source_commit"] = revision.stdout.strip() or None
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["resume_restores"] = "model, optimizer, EMA, step; RNG and data iterator restart from configured seed"
    name = "resume_config.json" if resuming else "resolved_config.json"
    (root / name).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    manifest_name = "resume_data_manifest.json" if resuming else "data_manifest.json"
    (root / manifest_name).write_text(json.dumps(dataset.manifest, indent=2) + "\n")


def archive_normalization(cfg, root, *, resuming):
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for phase in ("move", "operate"):
        target = assets / f"{phase}.json"
        if not resuming:
            source = Path(getattr(cfg, f"{phase}_norm_stats_path")).resolve()
            if source != target.resolve():
                shutil.copyfile(source, target)
        setattr(cfg, f"{phase}_norm_stats_path", str(target.resolve()))


def resolve_resume_config(cfg: WideCameraTrainConfig) -> WideCameraTrainConfig:
    path = Path(cfg.checkpoint_base_dir) / "wide_camera_dual_expert" / cfg.exp_name / "resolved_config.json"
    if not cfg.resume or cfg.overwrite or not path.exists():
        return cfg
    saved = json.loads(path.read_text())
    # A continuation keeps the original model/data/normalization/optimizer
    # contract. Operational settings and the total training budget may change.
    operational = {"exp_name", "checkpoint_base_dir", "num_steps", "num_workers", "log_interval",
                   "save_interval", "keep_period", "resume", "overwrite", "wandb_enabled", "fsdp_devices"}
    updates = {field.name: saved[field.name] for field in dataclasses.fields(cfg)
               if field.name not in operational}
    updates["task_names"] = tuple(updates["task_names"])
    updates["split_names"] = tuple(updates["split_names"])
    changed = [key for key, value in updates.items() if getattr(cfg, key) != value]
    if changed:
        print(f"Resuming with saved settings for: {', '.join(changed)}. Source: {path}")
    return dataclasses.replace(cfg, **updates)


def main(cfg: WideCameraTrainConfig) -> None:
    cfg = resolve_resume_config(cfg)
    train_config = _create_train_config(cfg)

    init_logging()
    random.seed(cfg.seed + jax.process_index())
    np.random.seed(cfg.seed + jax.process_index())
    torch.manual_seed(cfg.seed + jax.process_index())
    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        train_config.checkpoint_dir,
        keep_period=train_config.keep_period,
        overwrite=train_config.overwrite,
        resume=train_config.resume,
    )
    init_wandb(train_config, resuming=resuming, enabled=train_config.wandb_enabled)

    archive_normalization(cfg, train_config.checkpoint_dir, resuming=resuming)
    dataset = create_dataset(cfg)
    record_run(cfg, train_config, dataset, resuming=resuming)

    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len, model_path=cfg.tokenizer_path)
    data_iterator = create_data_iterator(cfg, tokenizer, data_sharding, dataset)
    batch = next(data_iterator)
    logging.info("Initialized data iterator:\n%s", training_utils.array_tree_to_info(batch))

    train_state, train_state_sharding = init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)

    ptrain_step = jax.jit(
        functools.partial(base_train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    def metrics_fn(state: training_utils.TrainState, batch, rng):
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        return model.compute_metrics(rng, batch[0], batch[1], train=True)

    pmetrics = jax.jit(
        metrics_fn,
        in_shardings=(train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    infos = []

    for step in range(start_step, train_config.num_train_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % train_config.log_interval == 0:
            stacked_infos = jax.tree.map(jnp.mean, common_utils.stack_forest(infos))
            info_host = jax.device_get(stacked_infos)
            infos = []

            metrics_rng, train_rng = jax.random.split(train_rng)
            metrics = jax.device_get(pmetrics(train_state, batch, metrics_rng))

            log_payload = {k: float(v) for k, v in info_host.items()}
            log_payload.update({k: float(v) for k, v in metrics.items()})
            log_payload["step"] = step

            if train_config.wandb_enabled:
                import wandb

                wandb.log(log_payload, step=step)

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in log_payload.items() if k != "step")
            print(f"Step {step}: {info_str}")

        batch = next(data_iterator)

        completed_steps = step + 1
        if completed_steps % train_config.save_interval == 0 or completed_steps == train_config.num_train_steps:
            _checkpoints.save_state(checkpoint_manager, train_state, completed_steps)

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(WideCameraTrainConfig))
