"""Shared model configuration for MTO training and inference."""

import dataclasses
from typing import Literal

from openpi.models import pi0_config


TrainingMode = Literal["full", "frozen_vlm", "lora", "lora_frozen_vlm"]


@dataclasses.dataclass(frozen=True)
class DualExpertPi0Config(pi0_config.Pi0Config):
    action_dim: int = 32
    action_horizon: int = 30
    max_token_len: int = 50


def create_mto_config(
    training_mode: TrainingMode = "full",
    *,
    action_dim: int = 32,
    action_horizon: int = 30,
    max_token_len: int = 50,
    dtype: str = "bfloat16",
    paligemma_variant: str | None = None,
    action_expert_variant: str | None = None,
) -> DualExpertPi0Config:
    """Choose backbone adaptation while keeping both heads and the router trainable."""
    vlm_variant, expert_variant, freeze_vlm = {
        "full": ("gemma_2b", "gemma_300m", False),
        "frozen_vlm": ("gemma_2b", "gemma_300m", True),
        "lora": ("gemma_2b_lora", "gemma_300m_lora", False),
        "lora_frozen_vlm": ("gemma_2b", "gemma_300m_lora", True),
    }[training_mode]
    return DualExpertPi0Config(
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        dtype=dtype,
        paligemma_variant=paligemma_variant or vlm_variant,
        action_expert_variant=action_expert_variant or expert_variant,
        freeze_paligemma=freeze_vlm,
    )
