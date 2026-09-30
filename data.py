import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.io import loadmat
from skimage.morphology import skeletonize
from torch.nn import functional as F
from torch.utils.data import Dataset

from .model import DIRECTIONS, shift
from .tiling import InputPolicy, padded_shape, view_arrays, windows

TARGET_VERSION = "native_unpruned_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Record:
    dataset: str
    sample_id: str
    image_path: Path
    mask_path: Path


def paired_records(dataset, root):
    root = Path(root)
    images = {p.stem: p for p in (root / "train").iterdir() if p.is_file()}
    masks = {p.stem: p for p in (root / ("train_lab" if dataset == "deepcrack" else "train_label")).iterdir()
             if p.is_file()}
    if images.keys() != masks.keys():
        raise ValueError("Image/label IDs do not match")
    return [Record(dataset, key, images[key], masks[key]) for key in sorted(images)]


def grouped_split(records, ratios, seed, key):
    groups = defaultdict(list)
    for record in records:
        groups[key(record.sample_id)].append(record)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))
    labels = ("train", "validation", "test")
    output = {label: [] for label in labels}
    targets = dict(zip(labels, (len(records) * value for value in ratios)))
    active = [name for name in labels if targets[name] > 0]
    for name in names:
        n = len(groups[name])
        label = min(active, key=lambda label: (
            (len(output[label]) + n - targets[label]) ** 2 - (len(output[label]) - targets[label]) ** 2,
            len(output[label]) / targets[label], labels.index(label)))
        output[label].extend(groups[name])
    return {key: sorted(value, key=lambda r: r.sample_id) for key, value in output.items()}


def build_splits(dataset, root, seed=3407):
    records = paired_records(dataset, root)
    if dataset in ("cfd", "cracktree260"):
        expected, train, val = (118, 82, 18) if dataset == "cfd" else (260, 182, 39)
        if len(records) != expected:
            raise ValueError(f"{dataset}: expected {expected} pairs, found {len(records)}")
        random.Random(seed).shuffle(records)
        return dict(train=records[:train], validation=records[train:train + val], test=records[train + val:])
    if dataset == "crack500":
        return grouped_split(records, (0.7, 0.15, 0.15), seed, lambda x: "_".join(x.split("_")[:2]))
    if dataset != "deepcrack":
        raise ValueError(f"Unknown dataset: {dataset}")
    def ids(name):
        return {Path(line.split()[0]).stem for line in (Path(root) / name).read_text().splitlines() if line.strip()}
    train, test = ids("train.txt"), ids("test.txt")
    by_id = {r.sample_id: r for r in records}
    pool = [by_id[key] for key in sorted(train - test)]
    split = grouped_split(pool, (0.9, 0.1, 0), seed, lambda x: re.sub(r"-\d+$", "", x))
    split["test"] = [by_id[key] for key in sorted(test)]
    return split


def split_ids(splits):
    return {name: [r.sample_id for r in records] for name, records in splits.items()}


def records_from_ids(dataset, root, ids):
    by_id = {r.sample_id: r for r in paired_records(dataset, root)}
    return {name: [by_id[key] for key in keys] for name, keys in ids.items()}


def load_pair(record):
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if record.mask_path.suffix.lower() == ".mat":
        annotations = np.atleast_1d(loadmat(record.mask_path, squeeze_me=True,
                                          struct_as_record=False)["groundTruth"]).ravel()
        mask = np.logical_or.reduce([np.asarray(a.Segmentation) == 2 for a in annotations])
    else:
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError(f"Unreadable or mismatched image/label: {record.sample_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (mask > 0).astype(np.uint8)


def target_arrays(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    centerline = skeletonize(mask > 0).astype(np.uint8)
    components, _ = ndimage.label(mask, np.ones((3, 3)))
    return dict(mask_native=mask, centerline=centerline, components=components.astype(np.int32))


class NativeStore:
    """Disk cache for original-size targets; small in-memory sampling indices."""
    def __init__(self, records, cache_dir):
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shapes, self.hashes, self.points = [], {}, []
        for record in records:
            image, mask = load_pair(record)
            digest = sha256(record.mask_path)
            path = self.path(record)
            current = False
            if path.exists():
                with np.load(path) as archive:
                    current = (archive["version"].item() == TARGET_VERSION
                               and archive["label_hash"].item() == digest
                               and archive["mask_native"].shape == mask.shape)
            if not current:
                np.savez_compressed(path, **target_arrays(mask), version=TARGET_VERSION, label_hash=digest)
            with np.load(path) as archive:
                skeleton = np.argwhere(archive["centerline"] > 0)
                comp = archive["components"]
            coords = np.argwhere(mask > 0)
            # Group all component pixels once, without repeated full-image scans.
            if len(coords):
                labels = comp[coords[:, 0], coords[:, 1]]
                order = np.argsort(labels, kind="stable")
                coords, labels = coords[order], labels[order]
                offsets = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
            else:
                offsets = np.array([0])
            self.points.append((coords, offsets, skeleton))
            self.shapes.append(image.shape[:2])
            self.hashes[record.sample_id] = digest

    def path(self, record):
        return self.cache_dir / f"{record.sample_id}.npz"

    def load(self, index):
        image, _ = load_pair(self.records[index])
        with np.load(self.path(self.records[index])) as archive:
            return image, archive["mask_native"].copy(), archive["centerline"].copy()


def tensor_targets(arrays):
    target = {key: torch.from_numpy(np.ascontiguousarray(value)).float()[None]
              for key, value in arrays.items()}
    mask = target["mask_native"][None]
    valid = target["valid_mask"][None]
    supervision = target["supervision_mask"][None]
    connections, weights = [], []
    for dy, dx in DIRECTIONS:
        connections.append(mask * shift(mask, dy, dx))
        weights.append(supervision * shift(valid, dy, dx))
    target["connection_target"] = torch.cat(connections, 1)[0]
    target["connection_weight"] = torch.cat(weights, 1)[0]
    for scale in (2, 4):
        area = F.avg_pool2d(valid, scale)
        target[f"mask_coverage_{scale}"] = (F.avg_pool2d(mask * valid, scale) / area.clamp_min(1e-8))[0]
        target[f"coverage_weight_{scale}"] = F.avg_pool2d(supervision, scale)[0]
    return target


class NativeDataset(Dataset):
    def __init__(self, store, policy, seed=3407, overfit=False):
        self.store, self.policy, self.seed, self.overfit = store, policy, seed, overfit
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        self.views, self.shapes = [], []
        for index, shape in enumerate(self.store.shapes):
            if self.overfit:
                origins = [None] if self.policy.full(shape) else windows(shape, self.policy)
            elif self.policy.full(shape):
                origins = [None] * 4
            else:
                coords, offsets, skeleton = self.store.points[index]
                origins = []
                for mode in range(4):
                    point = None
                    if mode == 2 and len(offsets) > 1:
                        component = rng.randrange(len(offsets) - 1)
                        point = coords[rng.randrange(int(offsets[component]), int(offsets[component + 1]))]
                    if mode == 3 and len(skeleton):
                        point = skeleton[rng.randrange(len(skeleton))]
                    if point is None:
                        origin = tuple(rng.randint(0, max(0, n - self.policy.core)) for n in shape)
                    else:
                        origin = tuple(rng.randint(max(0, int(p) - self.policy.core + 1),
                                                   min(int(p), max(0, n - self.policy.core)))
                                       for p, n in zip(point, shape))
                    origins.append(origin)
            for origin in origins:
                rotation = 0 if self.overfit else rng.randrange(4)
                horizontal = False if self.overfit else rng.random() < 0.5
                vertical = False if self.overfit else rng.random() < 0.5
                gain = 1 if self.overfit else rng.uniform(0.8, 1.2)
                bias = 0 if self.overfit else rng.uniform(-0.05, 0.05)
                self.views.append((index, origin, rotation, horizontal, vertical, gain, bias))
                dimensions = padded_shape(shape) if origin is None else (self.policy.tile_size,) * 2
                self.shapes.append(dimensions[::-1] if rotation % 2 else dimensions)

    def batches(self, batch_size, epoch):
        buckets = defaultdict(list)
        for i, shape in enumerate(self.shapes):
            buckets[shape].append(i)
        rng = random.Random(self.seed + epoch)
        batches = []
        for indices in buckets.values():
            rng.shuffle(indices)
            batches.extend(indices[i:i + batch_size] for i in range(0, len(indices), batch_size))
        rng.shuffle(batches)
        return batches

    def __len__(self):
        return len(self.views)

    def __getitem__(self, item):
        index, origin, rotation, horizontal, vertical, gain, bias = self.views[item]
        rgb, mask, skeleton = self.store.load(index)
        rgb, arrays, meta = view_arrays(rgb, mask, skeleton, self.policy, origin)
        def transform(value):
            if horizontal:
                value = np.flip(value, 1)
            if vertical:
                value = np.flip(value, 0)
            return np.ascontiguousarray(np.rot90(value, rotation))
        image = torch.from_numpy(transform(rgb)).permute(2, 0, 1).float() / 255
        image = (image * gain + bias).clamp(0, 1)
        target = tensor_targets({key: transform(value) for key, value in arrays.items()})
        meta["augmentation"] = (rotation, horizontal, vertical)
        return {"image": image, "targets": target, "meta": meta,
                "sample_id": self.store.records[index].sample_id}


def collate(samples):
    return {"image": torch.stack([s["image"] for s in samples]),
            "targets": {key: torch.stack([s["targets"][key] for s in samples]) for key in samples[0]["targets"]},
            "metas": [s["meta"] for s in samples], "sample_ids": [s["sample_id"] for s in samples]}
