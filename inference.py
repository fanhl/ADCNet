import time

import numpy as np
import torch

from .tiling import Stitcher, view_arrays, windows


def image_tensor(rgb, device):
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 255


@torch.no_grad()
def predict_native(model, image, policy, device, *, amp=False, stitching="gaussian"):
    """Unlabelled RGB -> original HxW probability, shared by validation and CLI."""
    if stitching not in Stitcher.MODES:
        raise ValueError(f"Unknown stitching mode {stitching!r}; expected one of {Stitcher.MODES}")
    model.eval()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    h, w = image.shape[:2]
    empty = np.zeros((h, w), np.uint8)
    seam = np.zeros((h, w), bool)
    def forward(rgb, arrays):
        valid = torch.from_numpy(arrays["valid_mask"])[None, None].to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            return model(image_tensor(rgb, device), valid).mask_logits.float().sigmoid()[0, 0].cpu().numpy()
    if policy.full((h, w)):
        rgb, arrays, _ = view_arrays(image, empty, empty, policy)
        probability = forward(rgb, arrays)[:h, :w]
        count = 1
    else:
        stitcher = Stitcher((h, w), policy, mode=stitching)
        origins = windows((h, w), policy)
        for y, x in origins:
            rgb, arrays, _ = view_arrays(image, empty, empty, policy, (y, x))
            result = forward(rgb, arrays)
            c, a = policy.core, policy.halo
            stitcher.add(result[a:a + c, a:a + c], (y, x))
            for edge in (y, y + c):
                if 0 < edge < h:
                    seam[max(0, edge - 2):min(h, edge + 2), x:min(w, x + c)] = True
            for edge in (x, x + c):
                if 0 < edge < w:
                    seam[y:min(h, y + c), max(0, edge - 2):min(w, edge + 2)] = True
        probability, count = stitcher.finish(), len(origins)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return probability, {"seconds": time.perf_counter() - started,
                         "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                         "windows": count, "mode": "full" if policy.full((h, w)) else "tiled",
                         "stitching": stitching,
                         "seam_mask": seam}
