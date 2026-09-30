import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .data import load_pair
from .inference import predict_native


def ratio(a, b, empty=1.0):
    return float(a / b) if b else float(empty)


def boundary_f1(prediction, target, tolerance):
    kernel = np.ones((3, 3), np.uint8)
    def boundary(mask):
        value = mask.astype(np.uint8)
        return value > cv2.erode(value, kernel)
    p, t = boundary(prediction), boundary(target)
    if not p.any() and not t.any():
        return 1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    precision = ratio((p & (cv2.dilate(t.astype(np.uint8), kernel) > 0)).sum(), p.sum(), 0)
    recall = ratio((t & (cv2.dilate(p.astype(np.uint8), kernel) > 0)).sum(), t.sum(), 0)
    return ratio(2 * precision * recall, precision + recall, 0)


def short_branches(skeleton, max_length=16):
    """8-neighbour terminal paths of <=16 pixels; diagnostic only, never pruning."""
    degree = ndimage.convolve(skeleton.astype(np.int16), np.ones((3, 3), np.int16), mode="constant") - skeleton
    result = np.zeros_like(skeleton)
    h, w = skeleton.shape
    for start in map(tuple, np.argwhere(skeleton & (degree <= 1))):
        path, seen, current = [], set(), start
        while len(path) <= max_length:
            if current in seen or (path and degree[current] >= 3):
                break
            path.append(current)
            seen.add(current)
            y, x = current
            neighbours = [(yy, xx) for yy in range(max(0, y - 1), min(h, y + 2))
                          for xx in range(max(0, x - 1), min(w, x + 2))
                          if skeleton[yy, xx] and (yy, xx) not in seen]
            if len(neighbours) != 1:
                break
            current = neighbours[0]
        if len(path) <= max_length:
            yy, xx = zip(*path)
            result[yy, xx] = True
    return result


def image_metrics(probability, target, threshold=0.5, seam=None):
    prediction, truth = probability >= threshold, target > 0
    tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    foreground_iou = ratio(tp, tp + fp + fn)
    background_iou = ratio(tn, tn + fp + fn)
    ps, ts = skeletonize(prediction), skeletonize(truth)
    top_p, top_r = ratio((ps & truth).sum(), ps.sum(), 0), ratio((ts & prediction).sum(), ts.sum(), 0)
    cldice = 1.0 if not ps.any() and not ts.any() else ratio(2 * top_p * top_r, top_p + top_r, 0)
    thin = ts & ((2 * ndimage.distance_transform_edt(truth) - 1) <= 2)
    short = short_branches(ts)
    seam = np.zeros_like(truth) if seam is None else seam
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "pixels": int(truth.size),
            "dice": ratio(2 * tp, 2 * tp + fp + fn), "iou": foreground_iou,
            "miou": 0.5 * (foreground_iou + background_iou),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "cldice": cldice,
            **{f"boundary_f1_{t}": boundary_f1(prediction, truth, t) for t in (1, 2, 3)},
            "thin_hits": int((prediction & thin).sum()), "thin_pixels": int(thin.sum()),
            "short_hits": int((prediction & short).sum()), "short_pixels": int(short.sum()),
            "seam_errors": int(((prediction != truth) & seam).sum()), "seam_pixels": int(seam.sum()),
            "seam_absolute_error": float((np.abs(probability - truth) * seam).sum())}


def summarize(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    total = lambda key: sum(row[key] for row in rows)
    tp, fp, fn, tn = (total(key) for key in ("tp", "fp", "fn", "tn"))
    foreground_iou = ratio(tp, tp + fp + fn)
    background_iou = ratio(tn, tn + fp + fn)
    report = {"dice": ratio(2 * tp, 2 * tp + fp + fn), "iou": foreground_iou,
              "miou": 0.5 * (foreground_iou + background_iou),
              "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
              **{key: float(np.mean([row[key] for row in rows]))
                 for key in ("cldice", "boundary_f1_1", "boundary_f1_2", "boundary_f1_3")}}
    for name in ("thin", "short"):
        report[f"{name}_skeleton_recall"] = ratio(total(f"{name}_hits"), total(f"{name}_pixels")) if total(f"{name}_pixels") else None
        report[f"{name}_pixels"] = total(f"{name}_pixels")
    report.update(seam_error_rate=ratio(total("seam_errors"), total("seam_pixels")) if total("seam_pixels") else None,
                  seam_mae=ratio(total("seam_absolute_error"), total("seam_pixels")) if total("seam_pixels") else None,
                  mean_image_seconds=float(np.mean([row["seconds"] for row in rows])),
                  peak_cuda_mb=max(row["peak_cuda_mb"] for row in rows), sample_count=len(rows))
    return report


def evaluate_records(model, records, policy, device, threshold=0.5, amp=False,
                     stitching="gaussian"):
    rows = []
    for record in records:
        image, mask = load_pair(record)
        probability, info = predict_native(model, image, policy, device, amp=amp,
                                           stitching=stitching)
        seam = info.pop("seam_mask")
        rows.append({"sample_id": record.sample_id, **info, **image_metrics(probability, mask, threshold, seam)})
    return summarize(rows), rows
