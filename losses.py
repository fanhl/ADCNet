import torch
from torch import nn
from torch.nn import functional as F


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def binary_loss(logits, target, weight, pos_weight=3):
    logits, target, weight = logits.float(), target.float(), weight.float()
    bce = weighted_mean(F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=logits.new_tensor(pos_weight)), weight)
    p = logits.sigmoid()
    axes = (1, 2, 3)
    dice = 1 - (2 * (p * target * weight).sum(axes) + 1) / ((p + target).mul(weight).sum(axes) + 1)
    active = (weight.sum(axes) > 0).float()
    return bce + weighted_mean(dice, active)


def soft_erode(image):
    return torch.minimum(-F.max_pool2d(-image, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-image, (1, 3), 1, (0, 1)))


def soft_skeleton(image, iterations=10):
    def opened(value):
        return F.max_pool2d(soft_erode(value), 3, 1, 1)
    skeleton = F.relu(image - opened(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - opened(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits, target, centerline, valid, supervision):
    probability = logits.float().sigmoid()
    # Keep the halo while computing morphology. Only the reductions use the core.
    skeleton = soft_skeleton(probability * valid)
    axes = (1, 2, 3)
    precision = (skeleton * target * supervision).sum(axes) / (skeleton * supervision).sum(axes).clamp_min(1e-8)
    recall = (centerline * probability * supervision).sum(axes) / (centerline * supervision).sum(axes).clamp_min(1e-8)
    loss = 1 - 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    active = ((centerline * supervision).sum(axes) > 0).float()
    return weighted_mean(loss, active)


def connection_loss(logits, target, weight):
    error = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    # Balance each sample/direction independently, including all-negative edges.
    axes = (2, 3)
    positive, negative = weight * target, weight * (1 - target)
    pos_count, neg_count = positive.sum(axes), negative.sum(axes)
    pos = (error * positive).sum(axes) / pos_count.clamp_min(1)
    neg = (error * negative).sum(axes) / neg_count.clamp_min(1)
    active = (pos_count > 0).float() + (neg_count > 0).float()
    return weighted_mean((pos + neg) / active.clamp_min(1), (active > 0).float())


class NativeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, output, target, progress):
        cfg, t = self.config, target
        weight = t["supervision_mask"]
        losses = {"native": binary_loss(output.mask_logits, t["mask_native"], weight, cfg["pos_weight"])}
        zero = output.mask_logits.sum() * 0
        if cfg["coverage"]:
            values = [weighted_mean(F.binary_cross_entropy_with_logits(
                getattr(output, f"coverage_{s}_logits").float(), t[f"mask_coverage_{s}"], reduction="none"),
                t[f"coverage_weight_{s}"]) for s in (2, 4)]
            losses["coverage"] = sum(values) / 2
        if cfg["centerline"]:
            losses["centerline"] = binary_loss(output.centerline_logits, t["centerline"], weight, 3)
        if cfg["connection"]:
            losses["connection"] = connection_loss(output.connection_logits, t["connection_target"], t["connection_weight"])
        if cfg["cldice"]:
            losses["cldice"] = cldice_loss(output.mask_logits, t["mask_native"], t["centerline"], t["valid_mask"], weight)
        losses["total"] = losses["native"] + min(1, max(0, progress)) * sum(
            (cfg[key] * losses.get(key, zero) for key in ("coverage", "centerline", "connection", "cldice")), zero)
        return losses
