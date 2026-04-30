from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class PreferenceDataConfig:
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    max_prompt_length: int = 256
    max_response_length: int = 512
    batch_size: int = 4
    num_workers: int = 4
    val_split_ratio: float = 0.05      # used when dataset has no built-in val split
    val_split_seed: int = 42
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


# ── Dataset format parsers ────────────────────────────────────────────────────

def _extract_ultrafeedback(example: Dict) -> Tuple[str, str, str]:
    """Parses argilla/ultrafeedback-binarized-preferences-cleaned rows.

    Actual columns (verified against the cached dataset):
      prompt      — the user instruction string
      chosen      — list of {role, content} dicts; last entry is the assistant response
      rejected    — same structure
    """
    instruction = example["prompt"]
    chosen_response = example["chosen"][-1]["content"]
    rejected_response = example["rejected"][-1]["content"]
    return instruction, chosen_response, rejected_response


def _extract_hh_rlhf(example: Dict) -> Tuple[str, str, str]:
    """Parses Anthropic/hh-rlhf rows (Human/Assistant turn strings)."""
    def _split(text: str) -> Tuple[str, str]:
        marker = "\n\nAssistant: "
        idx = text.rfind(marker)
        if idx == -1:
            return text, ""
        return text[: idx + len(marker)], text[idx + len(marker):]

    prompt, chosen_response = _split(example["chosen"])
    _, rejected_response = _split(example["rejected"])
    return prompt, chosen_response, rejected_response


def _get_extractor(dataset_name: str):
    if "hh-rlhf" in dataset_name or "hh_rlhf" in dataset_name:
        return _extract_hh_rlhf
    return _extract_ultrafeedback


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize_pair(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
    max_prompt_length: int,
    max_response_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a single instruction+response pair.
    Labels mask the prompt tokens with -100 so loss is computed over the response only.
    """
    max_length = max_prompt_length + max_response_length

    # Build prompt text using chat template when available (e.g. Qwen2.5).
    messages = [{"role": "user", "content": instruction}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = f"User: {instruction}\nAssistant: "

    # Tokenize prompt alone to measure its token length after truncation.
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # Tokenize the full sequence.
    full_text = prompt_text + response + tokenizer.eos_token
    encoded = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)

    labels = input_ids.clone()
    labels[:prompt_len] = -100          # mask prompt tokens
    labels[attention_mask == 0] = -100  # mask padding tokens

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Dataset ───────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer: PreTrainedTokenizerBase,
        config: PreferenceDataConfig,
        extractor,
    ) -> None:
        self.data = hf_split
        self.tokenizer = tokenizer
        self.config = config
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        instruction, chosen_response, rejected_response = self.extractor(self.data[index])

        chosen = _tokenize_pair(
            self.tokenizer, instruction, chosen_response,
            self.config.max_prompt_length, self.config.max_response_length,
        )
        rejected = _tokenize_pair(
            self.tokenizer, instruction, rejected_response,
            self.config.max_prompt_length, self.config.max_response_length,
        )

        return {
            "chosen_input_ids":        chosen["input_ids"],
            "chosen_attention_mask":   chosen["attention_mask"],
            "chosen_labels":           chosen["labels"],
            "rejected_input_ids":      rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels":         rejected["labels"],
        }


# ── Public factory ────────────────────────────────────────────────────────────

def build_preference_dataloaders(
    config: PreferenceDataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, DataLoader]:
    extractor = _get_extractor(config.dataset_name)

    raw = load_dataset(config.dataset_name)
    available_splits = list(raw.keys())

    if "train" in available_splits and len(available_splits) == 1:
        # Dataset has only a train split — carve out a validation portion.
        split = raw["train"].train_test_split(
            test_size=config.val_split_ratio,
            seed=config.val_split_seed,
        )
        train_hf = split["train"]
        val_hf = split["test"]
    else:
        # Use whatever val split name the dataset provides (test / validation).
        val_split_name = next(
            (s for s in ("validation", "test") if s in available_splits), available_splits[-1]
        )
        train_hf = raw["train"]
        val_hf = raw[val_split_name]

    if config.max_train_samples is not None:
        train_hf = train_hf.select(range(min(config.max_train_samples, len(train_hf))))
    if config.max_val_samples is not None:
        val_hf = val_hf.select(range(min(config.max_val_samples, len(val_hf))))

    train_dataset = PreferenceDataset(train_hf, tokenizer, config, extractor)
    val_dataset = PreferenceDataset(val_hf, tokenizer, config, extractor)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)

    return {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    }
