from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from part1_implementation.model import SentimentClassifier, ModelOutputs


# -----------------------
# Naming
# -----------------------
def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s.strip())
    s = s[:max_len].strip("_")
    return s or "example"


# -----------------------
# Training curves (if available)
# -----------------------
def plot_training_curves(train_history_path: Path, out_dir: Path) -> None:
   
    if not train_history_path.exists():
        return

    payload = json.loads(train_history_path.read_text(encoding="utf-8"))
    history = payload.get("history", payload if isinstance(payload, list) else [])
    if not history:
        return

    df = pd.DataFrame(history)
    if "epoch" not in df.columns:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    if "dev_loss" in df.columns:
        ax.plot(df["epoch"], df["dev_loss"], label="dev_loss")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="val_loss")
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.24, 0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=200)
    plt.close(fig)

    # Accuracy
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    if "dev_acc" in df.columns:
        ax.plot(df["epoch"], df["dev_acc"], label="dev_acc")
    if "val_acc" in df.columns:
        ax.plot(df["epoch"], df["val_acc"], label="val_acc")
    ax.set_title("Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.88, 0.95)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_acc.png", dpi=200)
    plt.close(fig)


# -----------------------
# Confusion matrix
# -----------------------
def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Slice bars
# -----------------------
def plot_slice_bars(slice_out: Dict[str, Any], out_path: Path) -> None:
    rows = []
    for k in ["negation_true", "negation_false", "contrast_true", "contrast_false"]:
        if k in slice_out:
            rows.append((k, slice_out[k]["accuracy"], slice_out[k]["n"]))

    lb = slice_out.get("length_buckets", {})
    for b, m in lb.items():
        rows.append((f"len_{b}", m["accuracy"], m["n"]))

    df = pd.DataFrame(rows, columns=["slice", "accuracy", "n"]).dropna(subset=["accuracy"])

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.bar(df["slice"].tolist(), df["accuracy"].astype(float).tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Slice Accuracy")
    ax.set_ylabel("Accuracy")
    xs = np.arange(len(df))
    ax.bar(xs, df["accuracy"].values)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["slice"].tolist(), rotation=45, ha="right", fontsize=8)


    for i, (acc, n) in enumerate(zip(df["accuracy"], df["n"])):
        ax.text(i, float(acc) + 0.01, f"n={int(n)}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Confidence histogram
# -----------------------
def plot_confidence_hist(conf: np.ndarray, correct: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    ax.hist(conf[correct == 1], bins=20, alpha=0.7, label="Correct")
    ax.hist(conf[correct == 0], bins=20, alpha=0.7, label="Incorrect")

    ax.set_title("Confidence Histogram")
    ax.set_xlabel("Max probability")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Calibration plot
# -----------------------
def plot_calibration(conf: np.ndarray, correct: np.ndarray, out_path: Path, n_bins: int = 15) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_accs.append(float(np.mean(correct[mask])))
        bin_confs.append(float(np.mean(conf[mask])))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(bin_confs, bin_accs, marker="o")
    ax.set_title("Calibration (Reliability Diagram)")
    ax.set_xlabel("Avg confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Attention visualization helpers
# -----------------------
def merge_wordpieces(tokens: List[str], weights: np.ndarray) -> Tuple[List[str], np.ndarray]:
    merged_tokens: List[str] = []
    merged_weights: List[float] = []

    for tok, w in zip(tokens, weights.tolist()):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            merged_tokens.append(tok)
            merged_weights.append(float(w))
            continue

        if tok.startswith("##") and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + tok[2:]
            merged_weights[-1] += float(w)
        else:
            merged_tokens.append(tok)
            merged_weights.append(float(w))

    return merged_tokens, np.array(merged_weights, dtype=np.float32)


def save_attention_heatmap_for_text(
    model: SentimentClassifier,
    tokenizer,
    text: str,
    true_label: int,
    pred_label: int,
    confidence: float,
    out_path: Path,
    max_len: int,
    layer_index: int,
) -> None:
    
    model.eval()
    enc = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out: ModelOutputs = model(input_ids=input_ids, attention_mask=attention_mask, return_attentions=True)

    attentions = out.attentions
    if attentions is None or len(attentions) == 0:
        return

    layer = attentions[layer_index]  
    layer0 = layer[0]                
    avg = layer0.mean(dim=0)         
    cls_to_tokens = avg[0, :]        


    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    weights = cls_to_tokens.detach().cpu().numpy()

    pad_id = tokenizer.pad_token_id
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token

    keep_idx = []
    for i, (tid, tok) in enumerate(zip(ids, tokens)):
        # drop padding
        if pad_id is not None and tid == pad_id:
            continue
        # drop special tokens for cleaner axes
        if tok in {cls_tok, sep_tok}:
            continue
        keep_idx.append(i)

    # Fallback: if filtering removed everything, at least drop pads
    if len(keep_idx) == 0:
        keep_idx = [i for i, tid in enumerate(ids) if (pad_id is None or tid != pad_id)]

    tokens = [tokens[i] for i in keep_idx]
    weights = weights[keep_idx]
  

    tokens_m, weights_m = merge_wordpieces(tokens, weights)

    fig = plt.figure(figsize=(max(8, len(tokens_m) * 0.35), 2.2))
    ax = fig.add_subplot(111)

    ax.imshow(weights_m[np.newaxis, :], aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(len(tokens_m)))
    ax.set_xticklabels(tokens_m, rotation=90, fontsize=8)

    layer_name = "layer0" if layer_index == 0 else "layerLast"
    title = f"Attention {layer_name} (CLS→tokens) | true={true_label} pred={pred_label} conf={confidence:.3f}"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
