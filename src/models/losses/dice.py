
from __future__ import annotations
import torch
import torch.nn as nn
import monai
import warnings
from typing import Callable, Sequence
from monai.networks import one_hot

from super_gradients.common.registry.registry import register_loss

from .focal_loss import MMSegFocalLoss


@register_loss()
class DiceMMSegFocalLoss(nn.Module):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses. Here we take the focal loss
    implemented in the mmseg lib.
    """
    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Callable | None = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            gamma: float = 2.0,
            alpha: float = 0.5,
            focal_weight: Sequence[float] | float | int | torch.Tensor | None = None,
            lambda_dice: float = 1.0,
            lambda_focal: float = 1.0,
    ):
        super().__init__()
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            focal_weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        self.dice = monai.losses.DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

        self.focal = MMSegFocalLoss(
            use_sigmoid=True,
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            class_weight=focal_weight,
        )

        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        focal_loss = self.focal(input, target.squeeze(1).long())

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        dice_loss = self.dice(input, target)

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss


if __name__ == "__main__":
    inputs = torch.randn(2, 9, 224, 224)
    target = torch.randint(0, 9, (2, 1, 224, 224))
    loss = DiceMMSegFocalLoss(softmax=True, to_onehot_y=True, squared_pred=True, reduction='mean', include_background=True,)
    output = loss(inputs, target)

    print(output)