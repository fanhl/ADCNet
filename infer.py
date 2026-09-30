
import argparse
import csv
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Also support running this file directly, including from an IDE.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

from test7.data import load_pair, paired_records, sha256
from test7.inference import predict_native
from test7.metrics import image_metrics
from test7.tiling import InputPolicy
from test7.train_cfd import load_checkpoint, resolve_device, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESULT_METRICS = ("precision", "recall", "dice", "miou", "cldice", "boundary_f1_1")
CSV_FIELDS = (
    "image", "status", "height", "width", "inference_seconds", "read_seconds",
    "export_seconds", "total_seconds", "mode", "windows", "peak_cuda_mb",
    "threshold", "crack_pixels", "crack_fraction", "components",
    "skeleton_pixels", *RESULT_METRICS, "ground_truth", "metrics_status",
    "output_dir", "error",
)


def ground_truth_index(manifest):
    """Return manifest-backed labels keyed by sample ID, or an unavailability reason."""
    config = manifest.get("config", {})
    dataset = config.get("data", {}).get("dataset")
    root = manifest.get("dataset_root")
    if not dataset or not root:
        return {}, "checkpoint manifest has no dataset_root"
    root = Path(root)
    if not root.is_dir():
        return {}, f"dataset root is unavailable: {root}"
    try:
        records = paired_records(dataset, root)
    except (OSError, ValueError) as error:
        return {}, f"cannot load dataset labels: {error}"
    return {record.sample_id: record for record in records}, None


def prediction_metrics(probability, image_path, manifest, records, unavailable_reason, threshold):
    """Calculate the requested per-image metrics when a verified label is available."""
    empty = {key: None for key in RESULT_METRICS}
    record = records.get(Path(image_path).stem)
    if record is None:
        reason = unavailable_reason or f"no ground truth matches sample ID: {Path(image_path).stem}"
        return {**empty, "ground_truth": None, "metrics_status": reason}
    expected_hash = manifest.get("label_sha256", {}).get(record.sample_id)
    if expected_hash is not None and sha256(record.mask_path) != expected_hash:
        return {**empty, "ground_truth": str(record.mask_path),
                "metrics_status": "ground-truth hash differs from the training manifest"}
    try:
        _, target = load_pair(record)
    except (OSError, ValueError, cv2.error) as error:
        return {**empty, "ground_truth": str(record.mask_path),
                "metrics_status": f"cannot load ground truth: {error}"}
    if target.shape != probability.shape:
        return {**empty, "ground_truth": str(record.mask_path),
                "metrics_status": f"shape mismatch: prediction={probability.shape}, ground_truth={target.shape}"}
    values = image_metrics(probability, target, threshold)
    return {**{key: values[key] for key in RESULT_METRICS},
            "ground_truth": str(record.mask_path), "metrics_status": "ok"}


def discover_images(source, output, recursive=True):
    """Preserve relative names; exclude our output tree on repeated scans."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("Output must not equal or contain the input path")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {source}")
    candidates = source.rglob("*") if recursive else source.iterdir()
    paths = sorted(p for p in candidates if p.is_file()
                   and p.suffix.lower() in IMAGE_EXTENSIONS
                   and not p.resolve().is_relative_to(output))
    if not paths:
        raise ValueError(f"No supported images found in: {source}")
    return source, paths


def read_rgb(path):
    """Decode through bytes so Windows paths containing Chinese work too."""
    data = np.fromfile(path, dtype=np.uint8)
    if not data.size:
        raise ValueError(f"Empty image: {path}")
    # Keep stored pixel coordinates rather than applying EXIF orientation.
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path, value, *, rgb=False):
    if rgb:
        value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", value)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def overview_panel(tiles):
    """Six labelled previews; the individual exports retain native resolution."""
    cells = []
    for label, rgb in tiles:
        h, w = rgb.shape[:2]
        scale = min(480 / w, 320 / h, 1.0)
        preview = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        cell = np.full((356, 480, 3), 24, np.uint8)
        cv2.putText(cell, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y, x = 36 + (320 - preview.shape[0]) // 2, (480 - preview.shape[1]) // 2
        cell[y:y + preview.shape[0], x:x + preview.shape[1]] = preview
        cells.append(cell)
    return np.concatenate((np.concatenate(cells[:3], 1), np.concatenate(cells[3:], 1)), 0)


def export_results_thickness(rgb, probability, output, threshold=0.5, alpha=0.5, thickness=1):
    """Export unfiltered masks; skeleton/contour images are diagnostics only.

    Args:
        thickness: 控制加粗宽度的像素核大小 (默认为 3，如果缩放比例非常小，可设置为 5 或 7)
    """
    if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
        raise ValueError("Prediction must be a finite original-size probability map")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Prediction must be in [0, 1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probability = probability.astype(np.float32)
    mask = probability >= threshold
    binary = mask.astype(np.uint8) * 255
    skeleton = skeletonize(mask)

    # ---------------- 关键修改：膨胀核定义与白色部分加粗 ----------------
    # 定义形态学膨胀结构元（圆核视觉效果比方核更平滑自然）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))

    # 1. 加粗二值掩膜 (白色部分)
    dilated_binary = cv2.dilate(binary, kernel, iterations=1)

    # 2. 加粗骨架线 (白色部分)
    skeleton_uint8 = skeleton.astype(np.uint8) * 255
    dilated_skeleton = cv2.dilate(skeleton_uint8, kernel, iterations=1)
    # -------------------------------------------------------------------

    heatmap = cv2.cvtColor(cv2.applyColorMap(np.rint(probability * 255).astype(np.uint8),
                                             cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    # 覆盖层也使用加粗后的掩膜，避免缩小时红色裂缝消失
    dilated_mask = dilated_binary > 0
    overlay = rgb.copy()
    overlay[dilated_mask] = np.rint((1 - alpha) * rgb[dilated_mask] + alpha * np.array([255, 0, 0])).astype(np.uint8)

    # 轮廓线同样加粗（thickness 参数由原来的 1 改为动态配置）
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = rgb.copy()
    cv2.drawContours(contour, contours, -1, (0, 255, 0), max(1, thickness // 2))

    # 骨架彩色叠加层使用加粗后的骨架
    skeleton_rgb = rgb.copy()
    skeleton_rgb[dilated_skeleton > 0] = (0, 255, 255)

    np.save(output / "probability.npy", probability, allow_pickle=False)
    # Fixed [0,1] -> [0,65535] mapping, never per-image min/max normalization.
    write_png(output / "probability.png", np.rint(probability * 65535).astype(np.uint16))

    # 保存加粗后的白色部分图像
    write_png(output / "mask.png", dilated_binary)
    write_png(output / "skeleton.png", dilated_skeleton)

    for name, value in (("heatmap", heatmap), ("overlay", overlay),
                        ("contours", contour), ("skeleton_overlay", skeleton_rgb)):
        write_png(output / f"{name}.png", value, rgb=True)

    panel = overview_panel([
        ("Original", rgb), ("Probability (blue=0, red=1)", heatmap),
        (f"Mask (dilated)", np.repeat(dilated_binary[..., None], 3, 2)),
        ("Crack overlay", overlay), ("Contours", contour), ("Skeleton", skeleton_rgb),
    ])
    write_png(output / "overview.png", panel, rgb=True)

    # 注意：统计特征（裂缝像素数、占比等）仍基于原始未膨胀的精确数据计算，保证分析准确
    return {"crack_pixels": int(mask.sum()), "crack_fraction": float(mask.mean()),
            "components": int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1),
            "skeleton_pixels": int(skeleton.sum())}


def export_results(rgb, probability, output, threshold=0.5, alpha=0.5):
    """Export unfiltered masks; skeleton/contour images are diagnostics only."""
    if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
        raise ValueError("Prediction must be a finite original-size probability map")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Prediction must be in [0, 1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probability = probability.astype(np.float32)
    mask = probability >= threshold
    binary = mask.astype(np.uint8) * 255
    skeleton = skeletonize(mask)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.rint(probability * 255).astype(np.uint8),
                                           cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask] = np.rint((1 - alpha) * rgb[mask] + alpha * np.array([255, 0, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = rgb.copy()
    cv2.drawContours(contour, contours, -1, (0, 255, 0), 1)
    skeleton_rgb = rgb.copy()
    skeleton_rgb[skeleton] = (0, 255, 255)
    np.save(output / "probability.npy", probability, allow_pickle=False)
    # Fixed [0,1] -> [0,65535] mapping, never per-image min/max normalization.
    write_png(output / "probability.png", np.rint(probability * 65535).astype(np.uint16))
    write_png(output / "mask.png", binary)
    write_png(output / "skeleton.png", skeleton.astype(np.uint8) * 255)
    for name, value in (("heatmap", heatmap), ("overlay", overlay),
                        ("contours", contour), ("skeleton_overlay", skeleton_rgb)):
        write_png(output / f"{name}.png", value, rgb=True)
    panel = overview_panel([
        ("Original", rgb), ("Probability (blue=0, red=1)", heatmap),
        (f"Mask (threshold={threshold:g})", np.repeat(binary[..., None], 3, 2)),
        ("Crack overlay", overlay), ("Contours", contour), ("Skeleton", skeleton_rgb),
    ])
    write_png(output / "overview.png", panel, rgb=True)
    return {"crack_pixels": int(mask.sum()), "crack_fraction": float(mask.mean()),
            "components": int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1),
            "skeleton_pixels": int(skeleton.sum())}


def run_inference(checkpoint, source, output=None, *, device="auto", threshold=0.5,
                  alpha=0.5, recursive=True, amp=None, cpu_threads=None, overwrite=False):
    """Load a train.py checkpoint once and return a JSON-serializable run report.

    ``inference_seconds`` uses predict_native's CUDA-synchronized timer and includes
    preprocessing, all windows, transfer to CPU and stitching. ``total_seconds``
    spans image read through visual export, excluding checkpoint load and report
    serialization. No warm-up is performed; first-image startup cost is included.
    Unreadable images are reported and skipped; model/export failures raise.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve() if output is not None else checkpoint.parent / "inference"
    root, paths = discover_images(source, output, recursive)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise FileExistsError(f"Output is not an empty directory: {output}; use --overwrite or a new path")
    device = resolve_device(device)
    model, saved = load_checkpoint(checkpoint, device)
    manifest = saved["manifest"]
    config = manifest["config"]
    truth_records, truth_unavailable = ground_truth_index(manifest)
    threads = config["training"].get("cpu_threads", 4) if cpu_threads is None else cpu_threads
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    policy = InputPolicy(**config["input"])
    use_amp = bool(config["training"].get("amp", False) if amp is None else amp) and device.type == "cuda"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "architecture": saved["architecture"], "completed_epoch": saved.get("completed_epoch"),
        "source": str(Path(source).resolve()), "output": str(output),
        "device": str(device), "amp": use_amp, "cpu_threads": threads,
        "threshold": threshold, "overlay_alpha": alpha, "recursive": recursive,
        "input_policy": asdict(policy), "postprocessing": "none; skeleton is a diagnostic export only",
        "training": {key: manifest.get(key) for key in ("seed", "split_seed", "splits", "initialization")},
        "training_config": config,
        "timing": {"inference_seconds": "preprocess + all forward passes + CPU transfer/stitching; CUDA synchronized",
                   "total_seconds": "read + inference + image exports; excludes model load and JSON/CSV serialization",
                   "warmup": "none; first-image startup cost included"},
        "images": [],
    }
    # Release optimizer/RNG tensors from the training checkpoint during inference.
    del saved
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {checkpoint.name} on {device}; {len(paths)} image(s); threshold={threshold:g}", flush=True)
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        relative = path.relative_to(root)
        row = {"image": str(path), "relative_path": relative.as_posix()}
        try:
            rgb = read_rgb(path)
        except (OSError, ValueError, cv2.error) as error:
            row.update(status="error", error=str(error), total_seconds=time.perf_counter() - started)
            report["images"].append(row)
            print(f"[{index}/{len(paths)}] ERROR {relative}: {error}", flush=True)
            continue
        read_seconds = time.perf_counter() - started
        probability, info = predict_native(model, rgb, policy, device, amp=use_amp)
        info.pop("seam_mask")
        inference_seconds = info.pop("seconds")
        # Include the original extension to avoid collisions: road.jpg/ != road.png/.
        image_output = output / "images" / relative
        export_started = time.perf_counter()
        statistics = export_results(rgb, probability, image_output, threshold, alpha)
        export_seconds = time.perf_counter() - export_started
        metrics = prediction_metrics(probability, path, manifest, truth_records,
                                     truth_unavailable, threshold)
        row.update(status="ok", height=rgb.shape[0], width=rgb.shape[1], threshold=threshold,
                   read_seconds=read_seconds, inference_seconds=inference_seconds,
                   export_seconds=export_seconds, total_seconds=time.perf_counter() - started,
                   output_dir=str(image_output), **info, **statistics, **metrics)
        write_json(image_output / "result.json", row)
        report["images"].append(row)
        print(f"[{index}/{len(paths)}] {relative} | {rgb.shape[1]}x{rgb.shape[0]} "
              f"| {info['mode']} windows={info['windows']} | inference={inference_seconds:.4f}s "
              f"| total={row['total_seconds']:.4f}s | crack={statistics['crack_fraction']:.2%}", flush=True)
    successful = [row for row in report["images"] if row["status"] == "ok"]
    report["summary"] = {
        "input_count": len(paths), "success_count": len(successful), "error_count": len(paths) - len(successful),
        "sum_inference_seconds": sum(row["inference_seconds"] for row in successful),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in successful])) if successful else None,
        "sum_image_total_seconds": sum(row["total_seconds"] for row in report["images"]),
    }
    write_json(output / "summary.json", report)
    with (output / "timings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["images"])
    print(f"Finished: {len(successful)}/{len(paths)} succeeded. Results: {output}", flush=True)
    return report


def run_inference_multi(checkpoint,source, output=None, *, device="auto", threshold=0.5,
                  alpha=0.5, recursive=True, amp=None, cpu_threads=None, overwrite=False):
    """Load a train.py checkpoint once and return a JSON-serializable run report.

    ``inference_seconds`` uses predict_native's CUDA-synchronized timer and includes
    preprocessing, all windows, transfer to CPU and stitching. ``total_seconds``
    spans image read through visual export, excluding checkpoint load and report
    serialization. No warm-up is performed; first-image startup cost is included.
    Unreadable images are reported and skipped; model/export failures raise.
    """
    #pt_dir = "/tmp/pycharm_pd/test7/artifacts/training/3408_2/cfd/"
    pt_dir = Path(checkpoint).resolve()
    count = 0
    for pt in Path(pt_dir).glob('*.pt'):
        checkpoint = os.path.join(pt_dir,pt)
        output = None
        count = count + 1
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        checkpoint = Path(checkpoint).resolve()
        output = Path(output).resolve() if output is not None else checkpoint.parent / "inference"/str(count)
        root, paths = discover_images(source, output, recursive)
        if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
            raise FileExistsError(f"Output is not an empty directory: {output}; use --overwrite or a new path")
        device = resolve_device(device)
        model, saved = load_checkpoint(checkpoint, device)
        manifest = saved["manifest"]
        config = manifest["config"]
        truth_records, truth_unavailable = ground_truth_index(manifest)
        threads = config["training"].get("cpu_threads", 4) if cpu_threads is None else cpu_threads
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        policy = InputPolicy(**config["input"])
        use_amp = bool(config["training"].get("amp", False) if amp is None else amp) and device.type == "cuda"
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "architecture": saved["architecture"], "completed_epoch": saved.get("completed_epoch"),
            "source": str(Path(source).resolve()), "output": str(output),
            "device": str(device), "amp": use_amp, "cpu_threads": threads,
            "threshold": threshold, "overlay_alpha": alpha, "recursive": recursive,
            "input_policy": asdict(policy), "postprocessing": "none; skeleton is a diagnostic export only",
            "training": {key: manifest.get(key) for key in ("seed", "split_seed", "splits", "initialization")},
            "training_config": config,
            "timing": {"inference_seconds": "preprocess + all forward passes + CPU transfer/stitching; CUDA synchronized",
                       "total_seconds": "read + inference + image exports; excludes model load and JSON/CSV serialization",
                       "warmup": "none; first-image startup cost included"},
            "images": [],
        }
        # Release optimizer/RNG tensors from the training checkpoint during inference.
        del saved
        output.mkdir(parents=True, exist_ok=True)
        print(f"Loaded {checkpoint.name} on {device}; {len(paths)} image(s); threshold={threshold:g}", flush=True)
        for index, path in enumerate(paths, 1):
            started = time.perf_counter()
            relative = path.relative_to(root)
            row = {"image": str(path), "relative_path": relative.as_posix()}
            try:
                rgb = read_rgb(path)
            except (OSError, ValueError, cv2.error) as error:
                row.update(status="error", error=str(error), total_seconds=time.perf_counter() - started)
                report["images"].append(row)
                print(f"[{index}/{len(paths)}] ERROR {relative}: {error}", flush=True)
                continue
            read_seconds = time.perf_counter() - started
            probability, info = predict_native(model, rgb, policy, device, amp=use_amp)
            info.pop("seam_mask")
            inference_seconds = info.pop("seconds")
            # Include the original extension to avoid collisions: road.jpg/ != road.png/.
            image_output = output / "images" / relative
            export_started = time.perf_counter()
            statistics = export_results_thickness(rgb, probability, image_output, threshold, alpha,2)
            export_seconds = time.perf_counter() - export_started
            metrics = prediction_metrics(probability, path, manifest, truth_records,
                                         truth_unavailable, threshold)
            row.update(status="ok", height=rgb.shape[0], width=rgb.shape[1], threshold=threshold,
                       read_seconds=read_seconds, inference_seconds=inference_seconds,
                       export_seconds=export_seconds, total_seconds=time.perf_counter() - started,
                       output_dir=str(image_output), **info, **statistics, **metrics)
            write_json(image_output / "result.json", row)
            report["images"].append(row)
            print(f"[{index}/{len(paths)}] {relative} | {rgb.shape[1]}x{rgb.shape[0]} "
                  f"| {info['mode']} windows={info['windows']} | inference={inference_seconds:.4f}s "
                  f"| total={row['total_seconds']:.4f}s | crack={statistics['crack_fraction']:.2%}", flush=True)
        successful = [row for row in report["images"] if row["status"] == "ok"]
        report["summary"] = {
            "input_count": len(paths), "success_count": len(successful), "error_count": len(paths) - len(successful),
            "sum_inference_seconds": sum(row["inference_seconds"] for row in successful),
            "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in successful])) if successful else None,
            "sum_image_total_seconds": sum(row["total_seconds"] for row in report["images"]),
        }
        write_json(output / "summary.json", report)
        with (output / "timings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["images"])
        print(f"Finished: {len(successful)}/{len(paths)} succeeded. Results: {output}", flush=True)
    #return report

def main(argv=None):
    parser = argparse.ArgumentParser(description="Native-resolution crack segmentation with visual results and per-image timing")
    parser.add_argument("--checkpoint", type=Path,  default="/tmp/pycharm_pd/test7/artifacts/training/3408/cracktree260/")
    parser.add_argument("--input", "--image", dest="source", type=Path,default='../data_test/cracktree260/6242.jpg', help="Image file or directory")
    #parser.add_argument("--input", "--image", dest="source", type=Path, default='/tmp/pycharm_pd/data/CrackTree260/train/', help="Image file or directory")
    #parser.add_argument("--checkpoint", type=Path,  default="/tmp/pycharm_pd/test7/artifacts/training/3409/deepcrack")
    #parser.add_argument("--input", "--image", dest="source", type=Path, default='/tmp/pycharm_pd/data_test/deepcrack', help="Image file or directory")
    #parser.add_argument("--checkpoint", type=Path,default="/tmp/pycharm_pd/test7/artifacts/training/3408/crack500/best_dice.pt")
    #parser.add_argument("--input", "--image", dest="source", type=Path,default='/tmp/pycharm_pd/data_test/crack500/20160222_082414_641_361.jpg',help="Image file or directory")
    #parser.add_argument("--checkpoint", type=Path,default="/tmp/pycharm_pd/test7/artifacts/training/cfd/best_dice.pt")
    #parser.add_argument("--input", "--image", dest="source", type=Path, default='../data_test/cfd/019.jpg',help="Image file or directory")
    #parser.add_argument("--input", "--image", dest="source", type=Path,default='/tmp/pycharm_pd/data/cfd/train/', help="Image file or directory")

    parser.add_argument("--output", type=Path, help="Default: checkpoint directory / inference")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5, help="Red overlay opacity [0,1]")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Default: checkpoint training setting; CUDA only")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace matching results in an existing output; other files are retained")
    args = parser.parse_args(argv)
    if args.checkpoint.is_dir():
        run_inference_multi(**vars(args))
        return 0
    report = run_inference(**vars(args))
    return 1 if report["summary"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
