"""Save and strictly resume MTO parameters, optimizer state, and EMA."""

import dataclasses
import logging

from etils import epath
import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming



def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int,
):
    with at.disable_typechecking():
        train_state, params = _split_params(state, to_pure_dict=True)
    checkpoint_manager.save(step, {"train_state": train_state, "params": {"params": params}})


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int | None = None,
) -> training_utils.TrainState:
    with at.disable_typechecking():
        train_state, params = _split_params(state, to_pure_dict=True)
        restored = checkpoint_manager.restore(
            step,
            items={"train_state": train_state, "params": {"params": params}},
        )
    return _merge_params(restored["train_state"], restored["params"], template=state)


def _split_params(
    state: training_utils.TrainState,
    *,
    to_pure_dict: bool,
) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    if to_pure_dict and isinstance(params, nnx.State):
        params = params.to_pure_dict()
    return train_state, params


def _merge_params(
    train_state: training_utils.TrainState,
    params: dict[str, at.Params],
    *,
    template: training_utils.TrainState,
) -> training_utils.TrainState:
    reference = template.ema_params if template.ema_params is not None else template.params
    pure = params["params"]
    at.check_pytree_equality(
        expected=reference.to_pure_dict(),
        got=pure,
        check_shapes=True,
        check_dtypes=True,
    )
    # Copy the NNX structure before replacing values so Param types and metadata
    # survive restoration without mutating the caller's initialization template.
    restored = jax.tree.map(lambda value: value, reference)
    restored.replace_by_pure_dict(pure)
    if template.ema_params is not None:
        return dataclasses.replace(train_state, ema_params=restored)
    return dataclasses.replace(train_state, params=restored)
