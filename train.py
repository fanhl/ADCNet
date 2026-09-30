import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from test7 import ARCHITECTURE
from test7.data import (NativeDataset, NativeStore, build_splits, collate, records_from_ids,
                   sha256, split_ids, TARGET_VERSION)
from test7.losses import NativeLoss
from test7.metrics import evaluate_records
from test7.model import NativeCrackNet
from test7.tiling import InputPolicy

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
VALIDATION_GROUP_SIZE = 5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    InputPolicy(**config["input"])
    if config["data"]["dataset"] not in ("cfd", "crack500", "deepcrack", "cracktree260"):
        raise ValueError("Unknown dataset")
    if config["data"]["batch_size"] < 1 or config["training"]["epochs"] < 1:
        raise ValueError("batch_size and epochs must be positive")
    if config["training"]["loss_ramp_epochs"] < 1:
        raise ValueError("loss_ramp_epochs must be positive")
    if any(not math.isfinite(v) or v < 0 for v in config["losses"].values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def optimizer_scheduler(model, training, total_steps, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    warmup = min(training["warmup_epochs"] * steps_per_epoch, max(0, total_steps - 1))
    minimum = training["min_learning_rate"] / training["learning_rate"]
    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        phase = min(1, (step - warmup) / max(1, total_steps - warmup))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * phase))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def checkpoint_state(model, optimizer, scheduler, scaler, manifest, epoch, updates, pixels, best, validation):
    return {"architecture": ARCHITECTURE, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "manifest": manifest,
            "completed_epoch": epoch, "updates": updates, "supervised_pixels": pixels, "best": best,
            "validation": validation, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model = NativeCrackNet(**checkpoint["manifest"]["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def run(config, *, device="auto", output_dir=None, resume=None, overfit_samples=0,
        stop_after_epoch=None):
    training, data = config["training"], config["data"]
    seed, epochs = int(training["seed"]), int(training["epochs"])
    split_seed = int(data["split_seed"])
    seed_all(seed)
    torch.set_num_threads(int(training["cpu_threads"]))
    cv2.setNumThreads(1)
    device = resolve_device(device)
    root = (PROJECT_ROOT / data["root"]).resolve()
    output = Path(output_dir or PACKAGE_ROOT / "artifacts" / "training" / data["dataset"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending_dir = output / ".pending_validation"
    checkpoint = None
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        old = checkpoint["manifest"]
        if old["config_sha256"] != config_hash(config) or old["dataset_root"] != str(root):
            raise ValueError("Resume configuration/dataset root mismatch")
        overfit_samples = old["overfit_samples"]
        splits = records_from_ids(data["dataset"], root, old["splits"])
    else:
        if any(output.glob("*.pt")) or (pending_dir.exists() and any(pending_dir.glob("*.pt"))):
            raise FileExistsError("Output already contains checkpoints; use --resume or another directory")
        split_path = output / "splits.json"
        if split_path.exists():
            saved = json.loads(split_path.read_text(encoding="utf-8"))
            if saved["split_seed"] != split_seed or saved["overfit_samples"] != overfit_samples:
                raise ValueError("Saved split seed/debug mode mismatch")
            splits = records_from_ids(data["dataset"], root, saved["splits"])
        else:
            splits = build_splits(data["dataset"], root, split_seed)
            if overfit_samples:
                if not 1 <= overfit_samples <= len(splits["train"]):
                    raise ValueError("Invalid overfit sample count")
                subset = splits["train"][:overfit_samples]
                splits = dict(train=subset, validation=subset, test=[])
        model = NativeCrackNet(**config["model"]).to(device)
    write_json(output / "splits.json", dict(split_seed=split_seed, overfit_samples=overfit_samples, splits=split_ids(splits)))
    print(f"prepare dataset={data['dataset']} splits={ {k: len(v) for k, v in splits.items()} }", flush=True)
    cache_dir = PACKAGE_ROOT / "artifacts" / "targets" / data["dataset"]
    store = NativeStore(splits["train"], cache_dir)
    label_hashes = {r.sample_id: sha256(r.mask_path) for values in splits.values() for r in values}
    source_hashes = {p.name: sha256(p) for p in PACKAGE_ROOT.glob("*.py")}
    if checkpoint and (checkpoint["manifest"]["label_sha256"] != label_hashes
                       or checkpoint["manifest"]["source_sha256"] != source_hashes):
        raise ValueError("Resume labels or source code changed")
    policy = InputPolicy(**config["input"])
    dataset = NativeDataset(store, policy, seed, bool(overfit_samples))
    # Rotation changes bucket counts, so calculate the actual epoch budgets.
    budgets = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        budgets.append(len(dataset.batches(data["batch_size"], epoch)))
    manifest = {"architecture": ARCHITECTURE, "config": config, "config_sha256": config_hash(config),
                "dataset_root": str(root), "splits": split_ids(splits), "seed": seed, "split_seed": split_seed,
                "overfit_samples": overfit_samples, "initialization": "random", "pretrained_sha256": None,
                "source_sha256": source_hashes, "label_sha256": label_hashes, "target_version": TARGET_VERSION,
                "input_policy": asdict(policy), "blend": {"sigma": policy.core / 4, "min_weight": 1e-3},
                "steps_per_epoch": budgets, "parameters": sum(p.numel() for p in model.parameters()),
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "opencv": cv2.__version__, "device": str(device)},
                "selection_metric": "native_dice", "updates": 0, "supervised_pixels": 0}
    optimizer, scheduler = optimizer_scheduler(model, training, sum(budgets), budgets[0])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training["amp"])
    criterion = NativeLoss(config["losses"])
    start = updates = pixels = 0
    best = {"dice": -1.0, "cldice": -1.0}
    history = []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, updates, pixels = checkpoint["completed_epoch"], checkpoint["updates"], checkpoint["supervised_pixels"]
        best = checkpoint["best"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
        history_path = Path(resume).parent / "history.json"
        if history_path.exists():
            history = [row for row in json.loads(history_path.read_text()) if row["epoch"] <= start]
    manifest.update(updates=updates, supervised_pixels=pixels)
    write_json(output / "manifest.json", manifest)

    def rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def pending_checkpoints():
        if not pending_dir.exists():
            return []
        candidates = []
        for path in pending_dir.glob("epoch_*.pt"):
            try:
                epoch_number = int(path.stem.removeprefix("epoch_"))
            except ValueError as error:
                raise ValueError(f"Invalid pending checkpoint name: {path.name}") from error
            candidates.append((epoch_number, path))
        return sorted(candidates)

    def validate_pending_group():
        nonlocal best
        candidates = pending_checkpoints()
        if not candidates:
            return
        candidate_epochs = [epoch_number for epoch_number, _ in candidates]
        group_start = ((candidate_epochs[0] - 1) // VALIDATION_GROUP_SIZE) * VALIDATION_GROUP_SIZE + 1
        if candidate_epochs != list(range(group_start, candidate_epochs[-1] + 1)):
            raise ValueError(f"Pending validation epochs are not contiguous: {candidate_epochs}")
        if len(candidates) > VALIDATION_GROUP_SIZE:
            raise ValueError(f"Pending validation group is too large: {candidate_epochs}")
        history_by_epoch = {row["epoch"]: row for row in history}
        missing_history = [epoch_number for epoch_number in candidate_epochs
                           if epoch_number not in history_by_epoch]
        if missing_history:
            raise ValueError(f"Pending checkpoints have no history rows: {missing_history}")

        reports = {}
        group_best = None
        improved = {}
        for epoch_number, path in candidates:
            candidate = torch.load(path, map_location="cpu", weights_only=False)
            if candidate["architecture"] != ARCHITECTURE:
                raise ValueError(f"Pending checkpoint architecture mismatch: {path}")
            model.load_state_dict(candidate["model"], strict=True)
            print(f"validation epoch={epoch_number} original_images={len(splits['validation'])}", flush=True)
            validation_started = time.perf_counter()
            report, rows = evaluate_records(model, splits["validation"], policy, device, 0.5,
                                            scaler.is_enabled())
            history_by_epoch[epoch_number]["seconds"] += time.perf_counter() - validation_started
            history_by_epoch[epoch_number]["validation"] = report
            reports[epoch_number] = report
            if group_best is None or report["dice"] > group_best[2]["dice"]:
                group_best = (path, epoch_number, report, rows)
            for key in best:
                if report[key] > best[key]:
                    best[key] = report[key]
                    improved[key] = (path, report)
            print(json.dumps(history_by_epoch[epoch_number]), flush=True)
            del candidate

        for key, (path, report) in improved.items():
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["best"] = dict(best)
            state["validation"] = report
            torch.save(state, output / f"best_{key}.pt")

        group_path, selected_epoch, selected_report, selected_rows = group_best
        group_state = torch.load(group_path, map_location="cpu", weights_only=False)
        group_state["best"] = dict(best)
        group_state["validation"] = selected_report
        group_end = candidate_epochs[-1]
        group_label = f"{group_start:04d}_{group_end:04d}"
        torch.save(group_state, output / f"best_dice_group_{group_label}.pt")
        write_json(output / f"validation_group_{group_label}.json", {
            "group_start_epoch": group_start,
            "group_end_epoch": group_end,
            "selected_epoch": selected_epoch,
            "selection_metric": "dice",
            "validation": selected_report,
            "images": selected_rows,
        })
        write_json(output / "validation_images.json", selected_rows)

        latest_epoch, latest_path = candidates[-1]
        latest_state = torch.load(latest_path, map_location="cpu", weights_only=False)
        latest_state["best"] = dict(best)
        latest_state["validation"] = reports[latest_epoch]
        latest_state["rng"] = rng_state()
        model.load_state_dict(latest_state["model"], strict=True)
        for filename in ("last.pt", "recovery.pt"):
            torch.save(latest_state, output / filename)
        write_json(output / "history.json", history)
        write_json(output / "manifest.json", manifest)
        cleanup_dir = output / f".validated_{group_label}_cleanup"
        pending_dir.rename(cleanup_dir)
        for _, path in candidates:
            (cleanup_dir / path.name).unlink()
        if not any(cleanup_dir.iterdir()):
            cleanup_dir.rmdir()

    existing_pending = pending_checkpoints()
    if existing_pending:
        pending_epochs = [epoch_number for epoch_number, _ in existing_pending]
        expected_start = ((pending_epochs[0] - 1) // VALIDATION_GROUP_SIZE) * VALIDATION_GROUP_SIZE + 1
        if pending_epochs != list(range(expected_start, start + 1)):
            raise ValueError(f"Pending checkpoints do not match resume epoch {start}: {pending_epochs}")
        if len(existing_pending) == VALIDATION_GROUP_SIZE or start == epochs:
            validate_pending_group()

    run_end = min(epochs, stop_after_epoch or epochs)
    for epoch in range(start, run_end):
        dataset.set_epoch(epoch)
        loader = DataLoader(dataset, batch_sampler=dataset.batches(data["batch_size"], epoch),
                            num_workers=data["workers"], collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        totals, successful, skipped, epoch_pixels = {}, 0, 0, 0
        started = time.perf_counter()
        for index, batch in enumerate(loader):
            image = batch["image"].to(device)
            target = {k: v.to(device) for k, v in batch["targets"].items()}
            progress = (epoch + index / len(loader)) / training["loss_ramp_epochs"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = criterion(model(image, target["valid_mask"]), target, progress)
            if not all(torch.isfinite(v).item() for v in losses.values()):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch + 1}, batch={index + 1}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError("Nonfinite gradient")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
                continue
            scheduler.step()
            updates += 1
            successful += 1
            count = int(target["supervision_mask"].sum())
            pixels += count
            epoch_pixels += count
            for key, value in losses.items():
                totals[key] = totals.get(key, 0) + float(value.detach())
            if index == 0 or (index + 1) % 20 == 0:
                print(f"epoch={epoch + 1}/{epochs} batch={index + 1}/{len(loader)} loss={float(losses['total'].detach()):.5f}", flush=True)
        if not successful:
            raise RuntimeError("Epoch has no successful optimizer update")
        manifest.update(updates=updates, supervised_pixels=pixels)
        state = checkpoint_state(model, optimizer, scheduler, scaler, manifest, epoch + 1,
                                 updates, pixels, best, None)
        pending_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, pending_dir / f"epoch_{epoch + 1:04d}.pt")
        for filename in ("last.pt", "recovery.pt"):
            torch.save(state, output / filename)
        row = {"epoch": epoch + 1, "updates": updates, "supervised_pixels": pixels,
               "epoch_supervised_pixels": epoch_pixels, "train": {k: v / successful for k, v in totals.items()},
               "lr": optimizer.param_groups[0]["lr"], "amp_skipped_batches": skipped,
               "seconds": time.perf_counter() - started, "validation": None}
        history.append(row)
        write_json(output / "history.json", history)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(row), flush=True)
        if (epoch + 1) % VALIDATION_GROUP_SIZE == 0 or epoch + 1 == epochs:
            validate_pending_group()
    return output


def main():
    parser = argparse.ArgumentParser(description="Native-resolution GCDNet training")
    parser.add_argument("--config", type=Path, default='./configs/cfd.yaml')
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path,default='./artifacts/training/3408_2/cfd')
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int, help="Pause at an epoch boundary without changing the scheduled budget")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    run(config, device=args.device, output_dir=args.output_dir, resume=args.resume,
        overfit_samples=args.overfit_samples, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
