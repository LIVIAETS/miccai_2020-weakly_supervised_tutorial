#!/usr/bin/env python3

from torch import einsum
import torch.nn.functional as F

from .utils import simplex, sset


class CrossEntropy():
    def __init__(self, idk, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = idk
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, weak_target):
        assert probs.shape == weak_target.shape
        assert simplex(probs)
        assert sset(weak_target, [0, 1])

        log_p = (probs[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


# ######## ------ Size loss function  (naive way) ---------- ###########
# --- This function will push the prediction to be close ot sizeGT ---#
class NaiveSizeLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = [1]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, input, bounds):
        assert simplex(input)

        B, K, H, W = input.shape
        assert bounds.shape == (B, K, 2)

        pred_size = einsum("bkwh->bk", input)[:, self.idk]

        upper_bounds = bounds[:, self.idk, 1]
        lower_bounds = bounds[:, self.idk, 0]
        assert (upper_bounds >= 0).all() and (lower_bounds >= 0).all()

        # size < upper <==> size - upper < 0
        # lower < size <==> lower - size < 0
        loss = F.relu(pred_size - upper_bounds) ** 2 + F.relu(lower_bounds - pred_size) ** 2
        loss /= (W * H)

        return loss / 100
