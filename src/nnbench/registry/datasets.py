"""Data loaders for CIFAR-10 and GLUE SST-2, with deterministic synthetic fixtures."""

from collections.abc import Sized
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class _SyntheticTextDataset(Dataset):
    def __init__(self, count: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, 1000, (count, 32), generator=generator)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.randint(0, 2, (count,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


class _TokenizedSST2(Dataset):
    def __init__(self, split: str, tokenizer_name: str, max_length: int) -> None:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("SST-2 requires the 'nlp' extra: pip install -e '.[nlp]'") from exc
        hf_split = "validation" if split in {"val", "valid", "validation"} else split
        self.rows = load_dataset("glue", "sst2", split=hf_split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        tokens = self.tokenizer(
            row["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = row.get("label", -1)
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def list_datasets() -> list[str]:
    """Return registered dataset names."""
    return ["cifar10", "sst2"]


def get_dataloader(
    name: str,
    split: str = "train",
    batch_size: int = 32,
    *,
    synthetic: bool = False,
    num_samples: int = 64,
    seed: int = 42,
    download: bool = False,
    tokenizer_name: str = "distilbert/distilbert-base-uncased",
    max_length: int = 128,
    **loader_options: Any,
) -> DataLoader:
    """Create a registered data loader; set ``synthetic`` for offline smoke runs."""
    if batch_size < 1 or num_samples < 1:
        raise ValueError("batch_size and num_samples must be positive integers")
    key = name.lower()
    if key == "cifar10":
        if synthetic:
            generator = torch.Generator().manual_seed(seed)
            images = torch.rand(num_samples, 3, 32, 32, generator=generator)
            labels = torch.randint(0, 10, (num_samples,), generator=generator)
            dataset: Dataset | Sized = TensorDataset(images, labels)
        else:
            try:
                from torchvision.datasets import CIFAR10
                from torchvision.transforms import Compose, Normalize, ToTensor
            except ImportError as exc:
                raise ImportError(
                    "CIFAR-10 requires torchvision; install the 'vision' extra."
                ) from exc
            dataset = CIFAR10(
                root="data",
                train=split == "train",
                download=download,
                transform=Compose(
                    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
                ),
            )
    elif key == "sst2":
        if synthetic:
            dataset = _SyntheticTextDataset(num_samples, seed)
        else:
            dataset = _TokenizedSST2(split, tokenizer_name, max_length)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Registered datasets: {', '.join(list_datasets())}"
        )
    if synthetic:
        loader_options.setdefault("shuffle", split == "train")
    return DataLoader(dataset, batch_size=batch_size, **loader_options)
