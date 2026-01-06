from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from part1_implementation.model import SentimentClassifier, build_tokenizer, device_for_torch, ModelOutputs
from part1_implementation.viz import (
    safe_filename,
    plot_confusion_matrix,
    plot_slice_bars,
    plot_confidence_hist,
    plot_calibration,
    plot_training_curves,
    save_attention_heatmap_for_text,
)

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NEGATION_RE = re.compile(r"\b(not|no|never|n't|none|nobody|nothing|nowhere|neither|nor)\b", re.IGNORECASE)
CONTRAST_RE = re.compile(r"\b(but|however|though|although|yet|nevertheless|nonetheless)\b", re.IGNORECASE)


@dataclass
class EvalConfig:
    output_dir: str

    hf_dataset: str
    text_col: str
    label_col: str

    pretrained_name: str
    max_seq_length: int

    # Model knobs
    dropout: float
    pooling: str

    # attention
    attention_examples_per_tag: int
    attention_tags: List[str]

    # evaluation
    batch_size: int = 64


def load_config(config_path: str) -> EvalConfig:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    att = cfg.get("attention", {})
    train = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    return EvalConfig(
        output_dir=str(cfg["project"]["output_dir"]),
        hf_dataset=str(cfg["data"]["hf_dataset"]),
        text_col=str(cfg["data"]["text_col"]),
        label_col=str(cfg["data"]["label_col"]),
        pretrained_name=str(model_cfg["pretrained_name"]),
        max_seq_length=int(model_cfg["max_seq_length"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "cls")),
        attention_examples_per_tag=int(att.get("examples_per_tag", 3)),
        attention_tags=list(att.get("tags", ["easy_pos", "easy_neg", "negation", "contrast", "failure"])),
        batch_size=int(train.get("eval_batch_size", 64)),
    )


# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="part1_implementation/config.yaml")
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint path. If empty: tries outputs/checkpoints/best.pt then last.pt",
    )

    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Split to evaluate. For SST-2, 'test' maps to HF split 'validation' (scored split).",
    )

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Override device selection",
    )
    p.add_argument("--no_attention", action="store_true", help="Skip attention visualizations")
    p.add_argument("--max_examples", type=int, default=0, help="If >0, limit eval to first N examples (debug)")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return device_for_torch()
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# -----------------------
# Paths
# -----------------------
def get_repo_out_root(output_dir: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]  
    return (repo_root / output_dir).resolve()


# -----------------------
# Checkpoint loading
# -----------------------
def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    raise KeyError("Could not find model weights in checkpoint (expected 'model' or 'model_state_dict').")


def _torch_load_compat(path: Path, device: torch.device):
   
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_model_from_checkpoint(
    pretrained_name: str,
    dropout: float,
    pooling: str,
    ckpt_path: Path,
    device: torch.device,
) -> SentimentClassifier:
    model = SentimentClassifier(pretrained_name=pretrained_name, dropout=dropout, pooling=pooling).to(device)
    ckpt = _torch_load_compat(ckpt_path, device)
    state = _extract_state_dict(ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_checkpoint(out_root: Path, ckpt_arg: str) -> Path:
    if ckpt_arg:
        p = Path(ckpt_arg)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    best = out_root / "checkpoints" / "best.pt"
    last = out_root / "checkpoints" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found. Looked for: {best} and {last}")


# -----------------------
# Prediction
# -----------------------
@torch.inference_mode()
def predict_all(
    model: SentimentClassifier,
    tokenizer,
    texts: List[str],
    max_len: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out: ModelOutputs = model(input_ids=input_ids, attention_mask=attention_mask, return_attentions=False)
        probs = out.probs.detach().cpu().numpy()
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        preds.append(pred)
        confs.append(conf)
        probs_all.append(probs)

    return np.concatenate(preds), np.concatenate(confs), np.concatenate(probs_all)


# -----------------------
# Slices + metrics
# -----------------------
def length_bucket(token_count: int) -> str:
    if token_count <= 4:
        return "<=4"
    if token_count <= 8:
        return "5-8"
    if token_count <= 16:
        return "9-16"
    if token_count <= 32:
        return "17-32"
    return "33+"


def add_slices(df: pd.DataFrame, tokenizer, text_col: str) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()

    df = df.copy()
    df["slice_negation"] = [bool(NEGATION_RE.search(t)) for t in texts]
    df["slice_contrast"] = [bool(CONTRAST_RE.search(t)) for t in texts]

    tok_lens = [len(tokenizer.tokenize(t)) for t in texts]
    df["token_len"] = tok_lens
    df["slice_len_bucket"] = [length_bucket(n) for n in tok_lens]
    return df


def _acc_f1(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {"n": 0, "accuracy": None, "f1": None}
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(df["true_label"], df["pred_label"])),
        "f1": float(f1_score(df["true_label"], df["pred_label"])),
    }


def slice_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["overall"] = _acc_f1(df)

    out["negation_true"] = _acc_f1(df[df["slice_negation"] == True])
    out["negation_false"] = _acc_f1(df[df["slice_negation"] == False])

    out["contrast_true"] = _acc_f1(df[df["slice_contrast"] == True])
    out["contrast_false"] = _acc_f1(df[df["slice_contrast"] == False])

    out["length_buckets"] = {}
    for b in ["<=4", "5-8", "9-16", "17-32", "33+"]:
        out["length_buckets"][b] = _acc_f1(df[df["slice_len_bucket"] == b])

    return out


# -----------------------
# Calibration metrics
# -----------------------
def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    n, k = probs.shape
    y_onehot = np.zeros((n, k), dtype=np.float32)
    y_onehot[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


# -----------------------
# Main
# -----------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_root = get_repo_out_root(cfg.output_dir)
    ckpt_path = resolve_checkpoint(out_root, args.ckpt)

    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {out_root}")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    hf_split = "validation" if args.split == "test" else args.split

    print(f"[INFO] Loading dataset from Hugging Face: {cfg.hf_dataset} split={args.split} (hf={hf_split})")
    ds = load_dataset(cfg.hf_dataset)
    split_ds = ds[hf_split]

    df = pd.DataFrame(split_ds)
    df.rename(columns={cfg.label_col: "true_label", cfg.text_col: "sentence"}, inplace=True)
    df["sentence"] = df["sentence"].astype(str)

    if args.max_examples and args.max_examples > 0:
        df = df.head(int(args.max_examples)).copy()
        print(f"[INFO] Debug mode: evaluating first N={len(df)} examples")

    tokenizer = build_tokenizer(cfg.pretrained_name)
    model = load_model_from_checkpoint(cfg.pretrained_name, cfg.dropout, cfg.pooling, ckpt_path, device)

    pred, conf, probs = predict_all(
        model=model,
        tokenizer=tokenizer,
        texts=df["sentence"].tolist(),
        max_len=cfg.max_seq_length,
        device=device,
        batch_size=cfg.batch_size,
    )

    df["pred_label"] = pred.astype(int)
    df["confidence"] = conf.astype(float)
    df["prob_0"] = probs[:, 0].astype(float)
    df["prob_1"] = probs[:, 1].astype(float)

    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report_txt = classification_report(y_true, y_pred)

    print(f"[RESULT] {args.split} accuracy: {acc:.4f}")
    print(f"[RESULT] {args.split} F1:       {f1:.4f}")
    print("[RESULT] Confusion matrix:", cm)
    print("\n[RESULT] Classification report:\n", report_txt)

    # Slices + calibration
    df = add_slices(df, tokenizer=tokenizer, text_col="sentence")
    slice_out = slice_metrics(df)

    correct = (df["true_label"] == df["pred_label"]).astype(int).to_numpy()
    brier = brier_score(probs=probs, y_true=y_true)
    ece = expected_calibration_error(conf=conf, correct=correct, n_bins=15)

    # Save predictions + misclassified
    pred_path = metrics_dir / f"{args.split}_predictions.csv"
    df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"[INFO] Wrote predictions -> {pred_path}")

    mis = df[df["true_label"] != df["pred_label"]].copy().sort_values("confidence", ascending=False)

    mis_path = metrics_dir / f"{args.split}_misclassified.csv"
    mis.to_csv(mis_path, index=False, encoding="utf-8")
    print(f"[INFO] Wrote misclassified examples -> {mis_path}")

    # Save metrics json
    metrics = {
        "split": args.split,
        "hf_split": hf_split,
        "checkpoint": str(ckpt_path),
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_txt,
        "brier_score": brier,
        "ece_15bins": ece,
        "slice_metrics": slice_out,
        "n_examples": int(len(df)),
        "max_seq_length": int(cfg.max_seq_length),
        "pooling": cfg.pooling,
        "dropout": cfg.dropout,
    }

    metrics_path = metrics_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote eval metrics -> {metrics_path}")

    # Figures
    plot_confusion_matrix(np.array(cm), figures_dir / f"{args.split}_confusion_matrix.png")
    plot_slice_bars(slice_out, figures_dir / f"{args.split}_slice_accuracy.png")
    plot_confidence_hist(conf=conf, correct=correct, out_path=figures_dir / f"{args.split}_confidence_hist.png")
    plot_calibration(conf=conf, correct=correct, out_path=figures_dir / f"{args.split}_calibration.png", n_bins=15)

    # Training curves 
    plot_training_curves(metrics_dir / "train_history.json", figures_dir)

    # Attention visualizations
    if not args.no_attention:
        k = int(cfg.attention_examples_per_tag)

        # build selection buckets
        df2 = df.copy()
        df2["correct"] = (df2["true_label"] == df2["pred_label"])

        picks: Dict[str, pd.DataFrame] = {}
        picks["easy_pos"] = df2[(df2["correct"] == True) & (df2["true_label"] == 1)].sort_values(
            "confidence", ascending=False
        ).head(k)
        picks["easy_neg"] = df2[(df2["correct"] == True) & (df2["true_label"] == 0)].sort_values(
            "confidence", ascending=False
        ).head(k)
        picks["negation"] = df2[df2["slice_negation"] == True].sort_values(
            ["correct", "confidence"], ascending=[False, False]
        ).head(k)
        picks["contrast"] = df2[df2["slice_contrast"] == True].sort_values(
            ["correct", "confidence"], ascending=[False, False]
        ).head(k)
        picks["failure"] = df2[df2["correct"] == False].sort_values("confidence", ascending=False).head(k)

        for tag in cfg.attention_tags:
            if tag not in picks:
                continue
            sub = picks[tag]
            for i, row in sub.reset_index(drop=True).iterrows():
                text = str(row["sentence"])
                true_label = int(row["true_label"])
                pred_label = int(row["pred_label"])
                confidence = float(row["confidence"])

                base = safe_filename(f"{tag}_{i:03d}_{text[:60]}")
                out_pathL = figures_dir / f"{args.split}_{base}_attn_last.png"

                save_attention_heatmap_for_text(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    true_label=true_label,
                    pred_label=pred_label,
                    confidence=confidence,
                    out_path=out_pathL,
                    max_len=cfg.max_seq_length,
                    layer_index=-1,
                )

        print(f"[INFO] Attention figures saved under: {figures_dir}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
