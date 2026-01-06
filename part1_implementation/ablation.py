from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def write_cfg(base_cfg_path: Path, out_cfg_path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    for section, section_overrides in overrides.items():
        cfg.setdefault(section, {})
        cfg[section].update(section_overrides)

    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def first_existing_json(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these metrics files exist:\n" + "\n".join(str(p) for p in paths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", type=str, default="part1_implementation/config.yaml")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    base_cfg = Path(args.base_config).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    ab_root = repo_root / "outputs" / "ablations"
    ab_root.mkdir(parents=True, exist_ok=True)

    # Ablations 
    experiments = [
        ("pooling_cls", {"model": {"pooling": "cls"}}),
        ("pooling_mean", {"model": {"pooling": "mean"}}),
        ("dropout_0.1", {"model": {"dropout": 0.1}}),
        ("dropout_0.0", {"model": {"dropout": 0.0}}),
    ]

    results = []
    for name, overrides in experiments:
        exp_dir = ab_root / name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = exp_dir / "config.yaml"

        overrides = dict(overrides)
        overrides.setdefault("project", {})
        overrides["project"]["output_dir"] = str(exp_dir)

        overrides.setdefault("train", {})
        overrides["train"]["save_checkpoints"] = False

        write_cfg(base_cfg, cfg_path, overrides)

        # Train 
        train_cmd = [
            "python",
            "-m",
            "part1_implementation.train",
            "--config",
            str(cfg_path),
            "--device",
            args.device,
        ]
        if args.amp and args.device == "cuda":
            train_cmd.append("--amp")
        run(train_cmd)

        
        metrics_candidates = [
            exp_dir / "metrics" / "test_metrics_from_train.json",  
            exp_dir / "metrics" / "test_metrics.json",             
        ]
        metrics_path = first_existing_json(metrics_candidates)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        # Normalize keys 
        out = {
            "experiment": name,
            "metrics_file": str(metrics_path),
            "test_acc": metrics.get("test_acc", metrics.get("accuracy")),
            "test_loss": metrics.get("test_loss"),
            "best_ckpt": metrics.get("best_ckpt", metrics.get("best_ckpt_path")),
        }
        results.append(out)

    out_path = ab_root / "ablation_summary.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] Wrote: {out_path}")


if __name__ == "__main__":
    main()
