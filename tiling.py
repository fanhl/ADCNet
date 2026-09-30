from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputPolicy:
    full_image_pixels: int = 327680
    tile_size: int = 480
    halo: int = 64
    stride: int = 264

    def __post_init__(self):
        if (self.full_image_pixels < 0 or self.halo < 0 or self.core < 8
                or self.tile_size % 8 or self.halo % 8
                or not 1 <= self.stride <= self.core):
            raise ValueError("Invalid native-input budget/tile/halo/stride")

    @property
    def core(self):
        return self.tile_size - 2 * self.halo

    def full(self, shape):
        h, w = padded_shape(shape)
        return h * w <= self.full_image_pixels


def padded_shape(shape):
    return tuple((int(n) + 7) // 8 * 8 for n in shape[:2])


def starts(length, core, stride):
    last = max(0, length - core)
    return sorted(set([*range(0, last + 1, stride), last]))


def windows(shape, policy):
    return [(y, x) for y in starts(shape[0], policy.core, policy.stride)
            for x in starts(shape[1], policy.core, policy.stride)]


def extract(array, top, left, height, width, *, replicate=False):
    """Integer crop; only the part outside the original image is padded."""
    h, w = array.shape[:2]
    ys, xs = np.arange(top, top + height), np.arange(left, left + width)
    out = array[np.clip(ys, 0, h - 1)[:, None], np.clip(xs, 0, w - 1)[None, :]].copy()
    if not replicate:
        valid = ((ys >= 0) & (ys < h))[:, None] & ((xs >= 0) & (xs < w))[None, :]
        out[~valid] = 0
    return out


def view_arrays(image, mask, skeleton, policy, origin=None):
    h, w = image.shape[:2]
    if origin is None:
        height, width = padded_shape((h, w))
        top = left = 0
        core_box = (0, 0, height, width)
    else:
        top, left = origin[0] - policy.halo, origin[1] - policy.halo
        height = width = policy.tile_size
        core_box = (policy.halo, policy.halo, policy.core, policy.core)
    rgb = extract(image, top, left, height, width, replicate=True)
    valid = extract(np.ones((h, w), np.float32), top, left, height, width)
    supervision = np.zeros((height, width), np.float32)
    y, x, ch, cw = core_box
    supervision[y:y + ch, x:x + cw] = 1
    target = {"mask_native": extract(mask, top, left, height, width),
              "centerline": extract(skeleton, top, left, height, width),
              "valid_mask": valid, "supervision_mask": supervision * valid}
    meta = {"native_shape": (h, w), "crop_origin": (top, left),
            "core_box": core_box, "input_shape": (height, width),
            "padding": (max(-top, 0), max(-left, 0),
                        max(top + height - h, 0), max(left + width - w, 0))}
    return rgb, target, meta


def gaussian_weight(core):
    axis = (np.arange(core, dtype=np.float32) - (core - 1) / 2) / (core / 4)
    return np.maximum(np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2)), 1e-3)


class Stitcher:
    """Reconstruct a native canvas from core predictions.

    ``gaussian`` is the production path.  ``uniform`` and ``hard`` are kept as
    explicit experimental controls: uniform gives every overlapping estimate
    equal mass, while hard assigns each output pixel to the first tile that
    covers it.  Keeping these choices here (instead of post-processing saved
    masks) ensures that every ablation uses the same coordinate system.
    """

    MODES = ("gaussian", "uniform", "hard")

    def __init__(self, shape, policy, mode="gaussian"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown stitching mode {mode!r}; expected one of {self.MODES}")
        self.total = np.zeros(shape, np.float32)
        self.mass = np.zeros(shape, np.float32)
        self.count = np.zeros(shape, np.uint16)
        self.policy = policy
        self.mode = mode
        self.weight = (gaussian_weight(policy.core) if mode == "gaussian"
                       else np.ones((policy.core, policy.core), np.float32))

    def add(self, probability, origin):
        y, x = origin
        h = min(self.policy.core, self.total.shape[0] - y)
        w = min(self.policy.core, self.total.shape[1] - x)
        v = self.weight[:h, :w]
        if self.mode == "hard":
            # A deterministic top-left raster order gives each pixel exactly
            # one owner, including the final edge-aligned windows.
            v = (self.mass[y:y + h, x:x + w] == 0).astype(np.float32)
        self.total[y:y + h, x:x + w] += probability[:h, :w] * v
        self.mass[y:y + h, x:x + w] += v
        self.count[y:y + h, x:x + w] += 1

    def finish(self):
        if not np.all(self.mass > 0):
            raise ValueError("Sliding windows did not cover the entire original image")
        return self.total / self.mass
