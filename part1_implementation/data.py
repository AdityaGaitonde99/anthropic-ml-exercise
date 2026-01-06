from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


@dataclass
class DataConfig:
    hf_dataset: str
    text_col: str
    label_col: str
    train_dev_split: float
    max_seq_length: int


def load_hf_splits(cfg: DataConfig) -> Tuple[Dataset, Dataset, Dataset]:
    
    dsd: DatasetDict = load_dataset(cfg.hf_dataset)

    train_full = dsd["train"]
    val_ds = dsd["validation"]

    train_ds, dev_ds = make_internal_dev_split(
        train_ds=train_full,
        label_col=cfg.label_col,
        dev_frac=cfg.train_dev_split,
        seed=0,  
    )

    return train_ds, dev_ds, val_ds


def make_internal_dev_split(
    train_ds: Dataset,
    label_col: str,
    dev_frac: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
   
    idx = np.arange(len(train_ds))
    y = np.array(train_ds[label_col])

    tr_idx, dev_idx = train_test_split(
        idx,
        test_size=dev_frac,
        random_state=seed,
        stratify=y,
    )
    return train_ds.select(tr_idx.tolist()), train_ds.select(dev_idx.tolist())


def tokenize_dataset(ds: Dataset, tokenizer, text_col: str, label_col: str, max_len: int) -> Dataset:
    
    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        enc = tokenizer(
            [str(x) for x in batch[text_col]],
            padding=False,
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = [int(x) for x in batch[label_col]]
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=list(ds.column_names))
    return ds


class HFPadCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]):
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")


def make_collate_fn(tokenizer) -> HFPadCollator:
    return HFPadCollator(tokenizer)
