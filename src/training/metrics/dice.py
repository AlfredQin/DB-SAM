import torch
import numpy as np
from torchmetrics import Metric
from typing import List

from super_gradients.common.registry.registry import register_metric

def compute_dice(mask_pred, mask_gt):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return 1
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum


@register_metric()
class DICEScore(Metric):
    def __init__(
            self,
            threshold: float = 0.5
    ):
        super().__init__()
        self.add_state("dice_score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: List[torch.Tensor], target: List[torch.Tensor], dataset_name=None) -> None:
        for pred, tgt in zip(preds, target):
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            pred = (pred > self.threshold).astype(np.uint8)

            tgt = tgt.detach().cpu().numpy()
            if dataset_name is not None and "2D" in dataset_name:
                dice_score = 0
                for i in range(pred.shape[0]):
                    dice_score += compute_dice(pred[i] > 0, tgt[i] > 0)
                self.dice_score += dice_score / pred.shape[0]
                self.total += 1
            else:
                self.dice_score += compute_dice(pred > 0, tgt > 0)
                self.total += 1

    def compute(self):
        return self.dice_score / self.total


@register_metric()
class MultiClassDICEScore(Metric):
    def __init__(self, threshold: float = 0.5, num_classes: int = 8):
        super().__init__()
        for i in range(1, num_classes + 1):
            self.add_state(f"dice_score_class_{i}", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        # self.add_state("dice_score", default=[{f"dice_score_class_{i}": torch.tensor(0, dtype=torch.float), "total": torch.tensor(0)} for i in range(1, num_classes + 1)], dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold
        self.num_classes = num_classes
        self.greater_component_is_better = {
            f"dice_score_class_{i}": True for i in range(1, num_classes + 1)
        }
        self.component_names = list(self.greater_component_is_better.keys())

    def update(self, preds: List[torch.Tensor], target: List[torch.Tensor]) -> None:
        """
        :param preds: list of tensors of shape (N, C, H, W), C = num_classes + 1
        :param target: list of tensors of shape (N, 1, H, W)
        :return:
        """
        for pred, tgt in zip(preds, target):   # pred: (N, C, H, W), tgt: (N, 1, H, W)
            pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)  # (N, H, W)
            pred = pred.detach().cpu().numpy()
            tgt = tgt.detach().squeeze(1).cpu().numpy()

            # compute dice score for each class
            for i in range(1, self.num_classes + 1):
                setattr(self, f"dice_score_class_{i}", getattr(self, f"dice_score_class_{i}") + self._compute_dice(pred == i, tgt == i))
            self.total += 1

    def compute(self):
        results = {
            f"dice_score_class_{i}": getattr(self, f"dice_score_class_{i}") / self.total for i in range(1, self.num_classes + 1)
        }

        return results

    def _compute_dice(self, pred, tgt):
        if pred.sum() > 0 and tgt.sum() > 0:
            return compute_dice(pred, tgt)
        elif pred.sum() == 0 and tgt.sum() == 0:
            return 1
        else:
            return 0


@register_metric()
class DICEScoreHQ(Metric):
    def __init__(
            self,
            threshold: float = 0.5
    ):
        super().__init__()
        self.add_state("dice_score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("dice_score_hq", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total_hq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold
        self.component_names = ["dice_score", "dice_score_hq"]

    def update(self, preds: List[List[torch.Tensor]], target: List[torch.Tensor]) -> None:
        preds, preds_hq = preds
        for pred, tgt in zip(preds, target):
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            pred = (pred > self.threshold).astype(np.uint8)

            tgt = tgt.detach().cpu().numpy()

            self.dice_score += compute_dice(pred > 0, tgt > 0)
            self.total += 1

        for pred, tgt in zip(preds_hq, target):
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            pred = (pred > self.threshold).astype(np.uint8)

            tgt = tgt.detach().cpu().numpy()

            self.dice_score_hq += compute_dice(pred > 0, tgt > 0)
            self.total_hq += 1

    def compute(self):
        dice = self.dice_score / self.total
        dice_hq = self.dice_score_hq / self.total_hq

        return {"dice_score": dice, "dice_score_hq": dice_hq}

if __name__ == "__main__":
    dice_score = MultiClassDICEScore()
    dice_score.update([torch.randn(1, 9, 256, 256)], [torch.randint(0, 9, (1, 1, 256, 256))])
    print()