from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from part1_implementation.data import make_internal_dev_split, tokenize_dataset, make_collate_fn
from part1_implementation.model import SentimentClassifier, build_tokenizer, device_for_torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Config
# -----------------------
@dataclass
class TrainConfig:
    seed: int
    output_dir: str

    hf_dataset: str
    text_col: str
    label_col: str
    train_dev_split: float

    pretrained_name: str
    max_seq_length: int

    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_ratio: float
    gradient_clip: float
    early_stopping_patience: int

    num_workers: int = 0
    label_smoothing: float = 0.0

    dropout: float = 0.1
    pooling: str = "cls" 

    save_checkpoints: bool = True
    save_infer_checkpoint: bool = True



def load_config(config_path: str) -> TrainConfig:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    return TrainConfig(
        seed=int(cfg["project"]["seed"]),
        output_dir=str(cfg["project"]["output_dir"]),
        hf_dataset=str(cfg["data"]["hf_dataset"]),
        text_col=str(cfg["data"]["text_col"]),
        label_col=str(cfg["data"]["label_col"]),
        train_dev_split=float(cfg["data"]["train_dev_split"]),
        pretrained_name=str(model_cfg["pretrained_name"]),
        max_seq_length=int(model_cfg["max_seq_length"]),
        batch_size=int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        num_epochs=int(train_cfg["num_epochs"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        gradient_clip=float(train_cfg["gradient_clip"]),
        early_stopping_patience=int(train_cfg["early_stopping_patience"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "cls")),
        save_checkpoints=bool(train_cfg.get("save_checkpoints", True)),
        save_infer_checkpoint=bool(train_cfg.get("save_infer_checkpoint", True))

    )


# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="part1_implementation/config.yaml")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint (best.pt or last.pt)")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Override device selection",
    )
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (torch>=2)")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return device_for_torch()
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# -----------------------
# Eval helper
# -----------------------
@torch.no_grad()
def evaluate_loss_acc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask, return_attentions=False)
        loss = ce(out.logits, labels)

        preds = torch.argmax(out.logits, dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_loss += float(loss.item()) * labels.size(0)
        total_n += int(labels.size(0))

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# -----------------------
# Checkpointing
# -----------------------
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_dev_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_dev_loss": best_dev_loss,
        },
        str(path),
    )

def save_infer_checkpoint(path: Path, model: torch.nn.Module, cfg: TrainConfig) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": {
            "hf_dataset": cfg.hf_dataset,
            "text_col": cfg.text_col,
            "label_col": cfg.label_col,
            "pretrained_name": cfg.pretrained_name,
            "max_seq_length": cfg.max_seq_length,
            "dropout": cfg.dropout,
            "pooling": cfg.pooling,
            "seed": cfg.seed,
        },
    }
    torch.save(payload, str(path))

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, float]:
    def _torch_load_compat(p: Path, dev: torch.device):
        try:
            return torch.load(str(p), map_location=dev, weights_only=False)
        except TypeError:
            return torch.load(str(p), map_location=dev)

    ckpt = _torch_load_compat(path, device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_dev_loss = float(ckpt.get("best_dev_loss", float("inf")))
    return start_epoch, best_dev_loss


# -----------------------
# Training runner
# -----------------------
def run_training(
    cfg: TrainConfig,
    resume_path: str = "",
    device_arg: str = "auto",
    use_amp: bool = False,
    use_compile: bool = False,
) -> Path:
    set_seed(cfg.seed)

    repo_root = Path(__file__).resolve().parents[1]  
    out_root = (repo_root / cfg.output_dir).resolve()
    ckpt_dir = out_root / "checkpoints"
    metrics_dir = out_root / "metrics"
    runs_dir = out_root / "runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_arg)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {out_root}")

    # Speed knobs (GPU)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[INFO] Loading dataset from Hugging Face: {cfg.hf_dataset}")
    print(f"[INFO] max_seq_length={cfg.max_seq_length} (token truncation limit)")
    print(f"[INFO] Dev split comes from HF train via train_dev_split={cfg.train_dev_split}")
    print(f"[INFO] Test split comes from HF validation (scored split)")
    print(f"[INFO] save_checkpoints={cfg.save_checkpoints}")

    ds = load_dataset(cfg.hf_dataset)
    train_ds = ds["train"]

    # Treat HF "validation" as TEST and evaluate it once at the end.
    test_ds = ds["validation"]

    train_train, train_dev = make_internal_dev_split(
        train_ds=train_ds,
        label_col=cfg.label_col,
        dev_frac=cfg.train_dev_split,
        seed=cfg.seed,
    )
    print(f"[INFO] train_train={len(train_train)} train_dev={len(train_dev)} test={len(test_ds)}")

    tokenizer = build_tokenizer(cfg.pretrained_name)

    # Pre-tokenize once
    train_train = tokenize_dataset(train_train, tokenizer, cfg.text_col, cfg.label_col, cfg.max_seq_length)
    train_dev = tokenize_dataset(train_dev, tokenizer, cfg.text_col, cfg.label_col, cfg.max_seq_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, cfg.text_col, cfg.label_col, cfg.max_seq_length)

    collate_fn = make_collate_fn(tokenizer)

    pin_memory = (device.type == "cuda")
    persistent_workers = (cfg.num_workers > 0)

    train_loader = DataLoader(
        train_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dev_loader = DataLoader(
        train_dev,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = SentimentClassifier(
        pretrained_name=cfg.pretrained_name,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  
            print("[INFO] Enabled torch.compile()")
        except Exception as e:
            print(f"[WARN] torch.compile() failed, continuing without it. error={e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    ce = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_dev_loss = float("inf")
    patience_left = cfg.early_stopping_patience
    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"
    infer_ckpt_path = ckpt_dir / "best_infer.pt"

    best_state_dict: Dict[str, torch.Tensor] | None = None

    # TensorBoard
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[INFO] TensorBoard logdir: {tb_dir}")

    
    start_epoch = 1
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_absolute():
            rp = (Path.cwd() / rp).resolve()
        if rp.exists():
            start_epoch, best_dev_loss = load_checkpoint(rp, model, optimizer, scheduler, device)
            patience_left = cfg.early_stopping_patience
            print(f"[INFO] Resumed from {rp} at epoch={start_epoch} best_dev_loss={best_dev_loss:.6f}")
        else:
            print(f"[WARN] Resume checkpoint not found: {rp} (starting fresh)")

    # AMP scaler
    use_amp = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, Any]] = []
    global_step = 0

    t0_total = time.time()

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        epoch_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.num_epochs}")

        running_loss = 0.0
        running_n = 0

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_attentions=False)
                loss = ce(out.logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_n += int(labels.size(0))

            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            global_step += 1

            pbar.set_postfix(train_loss=running_loss / max(running_n, 1))

        
        dev_loss, dev_acc = evaluate_loss_acc(model, dev_loader, device, label_smoothing=cfg.label_smoothing)

        epoch_seconds = time.time() - epoch_t0
        row = {
            "epoch": epoch,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        writer.add_scalar("dev/loss", dev_loss, epoch)
        writer.add_scalar("dev/acc", dev_acc, epoch)
        writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)

        print(
            f"[INFO] epoch={epoch} "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} "
            f"epoch_seconds={epoch_seconds:.1f}"
        )

        # Save last.pt 
        if cfg.save_checkpoints:
            save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_dev_loss)
            print(f"[INFO] Saved last checkpoint -> {last_ckpt_path}")

        # Save best by dev loss
        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            patience_left = cfg.early_stopping_patience

            # Always keep best weights in RAM 
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if cfg.save_checkpoints:
                save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_dev_loss)
                print(f"[INFO] Saved best checkpoint -> {best_ckpt_path}")
            else:
                print("[INFO] Best dev improved (checkpoints disabled; kept best weights in-memory)")

            if getattr(cfg, "save_infer_checkpoint", True):
                save_infer_checkpoint(infer_ckpt_path, model, cfg)
                print(f"[INFO] Saved inference checkpoint -> {infer_ckpt_path}")

        else:
            patience_left -= 1
            print(f"[INFO] No improvement. Early-stopping patience left: {patience_left}")
            if patience_left <= 0:
                print("[INFO] Early stopping triggered.")
                break

    total_seconds = time.time() - t0_total

    # Final TEST evaluation 
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # Load best before test eval:
    # - if checkpoints enabled: load best.pt
    # - else: load in-memory best_state_dict
    def _torch_load_compat(p: Path, dev: torch.device):
        try:
            return torch.load(str(p), map_location=dev, weights_only=False)
        except TypeError:
            return torch.load(str(p), map_location=dev)

    if cfg.save_checkpoints and best_ckpt_path.exists():
        best_obj = _torch_load_compat(best_ckpt_path, device)
        model.load_state_dict(best_obj["model"])
    else:
        if best_state_dict is None:
            raise RuntimeError("checkpoints disabled but best_state_dict is None — cannot run final test eval.")
        model.load_state_dict(best_state_dict)

    model.to(device)
    model.eval()

    test_loss, test_acc = evaluate_loss_acc(model, test_loader, device, label_smoothing=cfg.label_smoothing)
    print(f"[RESULT] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    test_metrics_path = metrics_dir / "test_metrics_from_train.json"
    test_payload = {
        "split": "test",
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_dev_loss": best_dev_loss,
        "best_ckpt": str(best_ckpt_path) if cfg.save_checkpoints else None,
        "save_checkpoints": bool(cfg.save_checkpoints),
    }
    test_metrics_path.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote test metrics -> {test_metrics_path}")

    # Write training history
    hist_path = metrics_dir / "train_history.json"
    payload = {
        "config": {
            "seed": cfg.seed,
            "hf_dataset": cfg.hf_dataset,
            "pretrained_name": cfg.pretrained_name,
            "max_seq_length": cfg.max_seq_length,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "num_epochs": cfg.num_epochs,
            "warmup_ratio": cfg.warmup_ratio,
            "gradient_clip": cfg.gradient_clip,
            "early_stopping_patience": cfg.early_stopping_patience,
            "num_workers": cfg.num_workers,
            "label_smoothing": cfg.label_smoothing,
            "dropout": cfg.dropout,
            "pooling": cfg.pooling,
            "save_checkpoints": cfg.save_checkpoints,
            "device": str(device),
            "amp": use_amp,
            "compile": use_compile,
        },
        "history": history,
        "total_seconds": total_seconds,
        "best_ckpt": str(best_ckpt_path),
        "last_ckpt": str(last_ckpt_path),
        "tensorboard_logdir": str(tb_dir),
        "final_test": {
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    }
    hist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote training history -> {hist_path}")
    print(f"[INFO] Total training seconds: {total_seconds:.1f}")

    writer.flush()
    writer.close()

    return best_ckpt_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("[TRAIN] main() started")
    best_ckpt = run_training(
        cfg,
        resume_path=args.resume,
        device_arg=args.device,
        use_amp=args.amp,
        use_compile=args.compile,
    )
    print(f"[TRAIN] done. best checkpoint at: {best_ckpt}")


if __name__ == "__main__":
    main()
