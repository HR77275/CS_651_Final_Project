from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


@dataclass
class LMConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch_dtype '{dtype_str}'. Choose from: {list(mapping)}")
    return mapping[dtype_str]


def _load_base_model(config: LMConfig) -> nn.Module:
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        return prepare_model_for_kbit_training(model)

    dtype = _resolve_dtype(config.torch_dtype)
    return AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)


def _attach_lora(model: nn.Module, config: LMConfig) -> nn.Module:
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def _build_ref_model(config: LMConfig) -> nn.Module:
    dtype = _resolve_dtype(config.torch_dtype)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    return ref_model


def build_lm_model(config: LMConfig) -> Tuple[nn.Module, nn.Module, AutoTokenizer]:
    """
    Returns (policy, ref_model, tokenizer).

    policy    — Qwen2.5 with LoRA/QLoRA adapters attached; only adapter params are trainable.
    ref_model — same base weights, fully frozen, no adapters; used for DPO reference log-probs.
    tokenizer — right-padded, pad_token set to eos_token if absent.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _load_base_model(config)
    policy = _attach_lora(base_model, config) if config.use_lora else base_model
    ref_model = _build_ref_model(config)

    return policy, ref_model, tokenizer
