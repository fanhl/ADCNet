from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import ARCHITECTURE


# (dy, dx): the reverse directions share the same undirected edge.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def shift(x, dy, dx):
    """At (y,x), read (y+dy,x+dx), with zero outside the image."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)))
    return padded[..., max(dy, 0):max(dy, 0) + h, max(dx, 0):max(dx, 0) + w]


def groups(channels):
    return next(n for n in (32, 16, 8, 4, 2, 1) if channels % n == 0)


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
                         nn.GroupNorm(groups(outgoing), outgoing), nn.GELU())


class AdaptiveDifferenceBlock(nn.Module):
    def __init__(self, channels, enabled=True, zero_sum=True, routing="dynamic"):
        super().__init__()
        if routing not in {"dynamic", "uniform"}:
            raise ValueError("routing must be 'dynamic' or 'uniform'")
        if not enabled and (not zero_sum or routing != "dynamic"):
            raise ValueError("zero_sum/routing ablations require enabled difference branches")
        self.enabled = enabled
        self.zero_sum = zero_sum
        self.routing = routing
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        if enabled:
            self.weight = nn.Parameter(torch.empty(5, channels, 1, 3, 3))
            nn.init.normal_(self.weight, std=0.12)
            support = torch.zeros(5, 1, 1, 3, 3)
            support[0] = 1
            support[1, ..., 1, :] = 1
            support[2, ..., :, 1] = 1
            support[3, ..., range(3), range(3)] = 1
            support[4, ..., range(3), [2, 1, 0]] = 1
            self.register_buffer("support", support)
            if routing == "dynamic":
                self.router = nn.Conv2d(channels, 6, 1)
        self.mix = nn.Sequential(nn.GroupNorm(groups(channels), channels), nn.GELU(),
                                 nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                                 nn.Conv2d(channels * 2, channels, 1))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def difference_weights(self):
        weight = self.weight * self.support
        if not self.zero_sum:
            return weight
        return weight - self.support * (weight.sum((-1, -2), keepdim=True)
                                        / self.support.sum((-1, -2), keepdim=True))

    def forward(self, x):
        local = self.local(x)
        if self.enabled:
            gates = self.router(x).softmax(1) if self.routing == "dynamic" else None
            local = local * (gates[:, :1] if gates is not None else 1 / 6)
            padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
            for i, weight in enumerate(self.difference_weights()):
                response = F.conv2d(padded, weight, groups=x.shape[1])
                local = local + (gates[:, i + 1:i + 2] * response
                                 if gates is not None else response / 6)
        return x + self.scale * self.mix(local)


class ConnectivityPropagation(nn.Module):
    def __init__(self, iterations=2):
        super().__init__()
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        self.iterations = iterations

    def forward(self, value, logits, valid):
        edges = logits.float().sigmoid()
        valid = valid.float()
        weights = [edges[:, i:i + 1] * valid * shift(valid, dy, dx)
                   for i, (dy, dx) in enumerate(DIRECTIONS)]
        for _ in range(self.iterations):
            total, mass = value.float(), torch.ones_like(valid)
            for edge, (dy, dx) in zip(weights, DIRECTIONS):
                reverse = shift(edge, -dy, -dx)
                total = total + edge * shift(value.float(), dy, dx) + reverse * shift(value.float(), -dy, -dx)
                mass = mass + edge + reverse
            value = (total / mass).to(value.dtype)
        return value


@dataclass
class NativeOutput:
    mask_logits: torch.Tensor
    base_logits: torch.Tensor
    centerline_logits: torch.Tensor
    connection_logits: torch.Tensor
    coverage_2_logits: torch.Tensor
    coverage_4_logits: torch.Tensor


class NativeCrackNet(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, channels=(24, 48, 96, 160), decoder_channels=48,
                 difference=True, propagation=True, propagation_channels=24,
                 propagation_iterations=2, difference_zero_sum=True,
                 difference_routing="dynamic"):
        super().__init__()
        if len(channels) != 4 or min(channels) < 4 or decoder_channels < 4:
            raise ValueError("Expected four encoder widths and decoder width >=4")
        if difference_routing not in {"dynamic", "uniform"}:
            raise ValueError("difference_routing must be 'dynamic' or 'uniform'")
        if not difference and (not difference_zero_sum or difference_routing != "dynamic"):
            raise ValueError("ADB formulation switches require difference=True")
        self.use_propagation = propagation
        self.encoder = nn.ModuleList()
        incoming = 3
        for i, width in enumerate(channels):
            self.encoder.append(nn.Sequential(ConvBlock(incoming, width, 1 if i == 0 else 2),
                                              AdaptiveDifferenceBlock(
                                                  width, difference, difference_zero_sum,
                                                  difference_routing),
                                              AdaptiveDifferenceBlock(
                                                  width, difference, difference_zero_sum,
                                                  difference_routing)))
            incoming = width
        self.projections = nn.ModuleList(nn.Conv2d(c, decoder_channels, 1) for c in channels)
        self.decoder = nn.ModuleList(nn.Sequential(ConvBlock(2 * decoder_channels, decoder_channels),
                                                   AdaptiveDifferenceBlock(
                                                       decoder_channels, difference,
                                                       difference_zero_sum, difference_routing))
                                     for _ in range(3))
        self.coverage_2 = nn.Conv2d(decoder_channels, 1, 1)
        self.coverage_4 = nn.Conv2d(decoder_channels, 1, 1)
        self.base = nn.Conv2d(decoder_channels, 1, 1)
        self.centerline = nn.Conv2d(decoder_channels, 1, 1)
        self.connection = nn.Conv2d(decoder_channels, 4, 1)
        self.reduction = nn.Conv2d(decoder_channels, propagation_channels, 1)
        self.propagation = ConnectivityPropagation(propagation_iterations)
        self.residual = nn.Sequential(ConvBlock(decoder_channels + propagation_channels, decoder_channels),
                                      nn.Conv2d(decoder_channels, 1, 1))

    def forward(self, image, valid_mask=None):
        features = []
        x = image
        for stage in self.encoder:
            x = stage(x)
            features.append(x)
        x = self.projections[3](features[3])
        decoded = {}
        for index in (2, 1, 0):
            x = F.interpolate(x, features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.decoder[index](torch.cat((x, self.projections[index](features[index])), 1))
            decoded[index] = x
        base, connection = self.base(x), self.connection(x)
        mask = base
        if self.use_propagation:
            valid = torch.ones_like(base) if valid_mask is None else valid_mask
            reduced = self.reduction(x)
            propagated = self.propagation(reduced, connection, valid)
            mask = base + 0.1 * self.residual(torch.cat((x, propagated - reduced), 1))
        return NativeOutput(mask, base, self.centerline(x), connection,
                            self.coverage_2(decoded[1]), self.coverage_4(decoded[2]))
