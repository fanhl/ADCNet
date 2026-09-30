import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from test7.data import records_from_ids, sha256
from test7.inference import predict_native
from test7.metrics import evaluate_records
from test7.tiling import InputPolicy
from test7.train_cfd import load_checkpoint, resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description="Original-size evaluation or unlabelled image inference")
    #parser.add_argument("--checkpoint", type=Path, default='/tmp/pycharm_pd/test7/artifacts/training/cracktree260/best_dice.pt')
    parser.add_argument("--checkpoint", type=Path,
                        default='/tmp/pycharm_pd/test7/artifacts/training/3408/cfd/best_dice.pt')
    #parser.add_argument("--checkpoint", type=Path,default='/tmp/pycharm_pd/test7/artifacts/training/crack500/best_dice.pt')
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    #parser.add_argument("--image", type=Path, help="Unlabelled single image or image directory")
    #parser.add_argument("--image", type=Path, default='../dataset/cracktree200/img')
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    #device = resolve_device(args.device)
    device = torch.device("cuda:5")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    manifest = checkpoint["manifest"]
    config = manifest["config"]
    torch.set_num_threads(config["training"]["cpu_threads"])
    policy = InputPolicy(**config["input"])
    if args.image:
        paths = [args.image] if args.image.is_file() else sorted(p for p in args.image.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        output = args.output or args.checkpoint.parent / "predictions"
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in paths:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            probability, info = predict_native(model, image, policy, device, amp=config["training"]["amp"])
            info.pop("seam_mask")
            np.save(output / f"{path.name}.probability.npy", probability)
            cv2.imwrite(str(output / f"{path.name}.mask.png"), (probability >= 0.5).astype(np.uint8) * 255)
            rows.append({"image": str(path), **info})
        write_json(output / "summary.json", rows)
    else:
        splits = records_from_ids(config["data"]["dataset"], manifest["dataset_root"], manifest["splits"])
        if any(sha256(r.mask_path) != manifest["label_sha256"][r.sample_id] for r in splits[args.split]):
            raise ValueError("Evaluation labels changed since training")
        report, rows = evaluate_records(model, splits[args.split], policy, device, amp=config["training"]["amp"])
        output = args.output or args.checkpoint.parent / f"{args.split}_evaluation.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, {"checkpoint": str(args.checkpoint.resolve()), "split": args.split, "threshold": 0.5,
                            "overfit_samples": manifest["overfit_samples"], "metrics": report, "images": rows})
        print(report)


if __name__ == "__main__":
    main()
