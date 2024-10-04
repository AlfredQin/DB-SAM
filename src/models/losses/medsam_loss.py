import monai
import torch.nn as nn

from super_gradients.common.registry.registry import register_loss


@register_loss()
class SegLoss(nn.Module):
    def __init__(
            self,
            sigmoid: bool = True,
            squared_pred: bool = True,
            reduction: str = 'mean',
            include_background: bool = True,
    ):
        super().__init__()
        self.seg_loss = monai.losses.DiceCELoss(
            include_background=include_background,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            reduction=reduction
        )

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.0
            for i in range(len(pred)):
                pred, target = pred[i].float(), target[i].float()
                loss += self.seg_loss(pred, target)
            return loss / len(pred)
        else:
            pred, target = pred.float(), target.float()   # in case of loss nan in mixed precision training
            return self.seg_loss(pred, target)


@register_loss()
class HQSegLoss(nn.Module):
    def __init__(
            self,
            sigmoid: bool = True,
            squared_pred: bool = True,
            reduction: str = 'mean',
            include_background: bool = True,
    ):
        super().__init__()
        self.seg_loss0 = SegLoss(
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            reduction=reduction,
            include_background=include_background
        )
        self.seg_loss1 = SegLoss(
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            reduction=reduction,
            include_background=include_background
        )

    def forward(self, pred, target):
        if isinstance(pred, tuple):
            return self.seg_loss0(pred[0], target) + self.seg_loss1(pred[1], target)
        else:
            return self.seg_loss0(pred, target)
