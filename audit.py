
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from .data import (NativeStore, build_splits, load_pair, split_ids, target_arrays, tensor_targets)
from .losses import NativeLoss
from .model import NativeCrackNet
from .tiling import InputPolicy, Stitcher, view_arrays, windows
from .train_cfd import PACKAGE_ROOT, PROJECT_ROOT, load_config, resolve_device, seed_all, write_json


def audit_dataset(name, samples=0, smoke=False, device="cpu"):
    config = load_config(PACKAGE_ROOT / "configs" / f"{name}.yaml")
    seed_all(config["training"]["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    splits = build_splits(name, PROJECT_ROOT / config["data"]["root"], config["data"]["split_seed"])
    records = [r for values in splits.values() for r in values]
    ids = [set(v) for v in split_ids(splits).values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    if name in ("crack500", "deepcrack"):
        import re
        key = (lambda x: "_".join(x.split("_")[:2])) if name == "crack500" else (lambda x: re.sub(r"-\d+$", "", x))
        families = [{key(x) for x in v} for v in ids]
        assert not families[0] & families[1]
        if name == "crack500":
            assert not (families[0] & families[2] or families[1] & families[2])
    chosen = records[:samples] if samples else records
    policy = InputPolicy(**config["input"])
    shapes, modes = Counter(), Counter()
    for record in chosen:
        image, mask = load_pair(record)
        shapes[str(mask.shape)] += 1
        modes["full" if policy.full(mask.shape) else "tiled"] += 1
        if policy.full(mask.shape):
            _, target, _ = view_arrays(image, mask, mask, policy)
            restored = target["mask_native"][:mask.shape[0], :mask.shape[1]]
        else:
            stitch = Stitcher(mask.shape, policy)
            for origin in windows(mask.shape, policy):
                _, target, _ = view_arrays(image, mask, mask, policy, origin)
                a, c = policy.halo, policy.core
                stitch.add(target["mask_native"][a:a + c, a:a + c], origin)
            restored = stitch.finish()
        np.testing.assert_array_equal(restored, mask)
    result = {"dataset": name, "splits": {k: len(v) for k, v in splits.items()}, "audited_images": len(chosen),
              "native_shapes_hw": dict(shapes), "input_modes": dict(modes), "label_roundtrip_dice": 1.0}
    if smoke:
        device = resolve_device(device)
        record = splits["train"][0]
        store = NativeStore([record], PACKAGE_ROOT / "artifacts" / "targets" / name)
        image, mask, skeleton = store.load(0)
        origin = None if policy.full(mask.shape) else windows(mask.shape, policy)[0]
        rgb, target, _ = view_arrays(image, mask, skeleton, policy, origin)
        target = {k: v[None].to(device) for k, v in tensor_targets(target).items()}
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        model = NativeCrackNet(**config["model"]).to(device)
        started = time.perf_counter()
        output = model(image_tensor, target["valid_mask"])
        losses = NativeLoss(config["losses"])(output, target, 1)
        losses["total"].backward()
        assert all(torch.isfinite(value).item() for value in losses.values())
        assert model.connection.weight.grad.abs().sum() > 0
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        path = PACKAGE_ROOT / "artifacts" / "smoke" / name / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": model.architecture, "model": model.state_dict()}, path)
        restored = NativeCrackNet(**config["model"]).to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(image_tensor, target["valid_mask"]).mask_logits,
                                       restored(image_tensor, target["valid_mask"]).mask_logits, rtol=0, atol=0)
        result["smoke"] = {"sample_id": record.sample_id, "input_shape": list(image_tensor.shape),
                           "losses": {k: float(v.detach()) for k, v in losses.items()},
                           "seconds": time.perf_counter() - started, "save_load_exact": True,
                           "parameters": sum(p.numel() for p in model.parameters())}
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", "cfd", "crack500", "cracktree260", "deepcrack"), default="all")
    parser.add_argument("--samples", type=int, default=0, help="0 audits all images")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "artifacts" / "audit.json")
    args = parser.parse_args()
    names = ("cfd", "crack500", "cracktree260", "deepcrack") if args.dataset == "all" else (args.dataset,)
    results = [audit_dataset(name, args.samples, args.smoke, args.device) for name in names]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
