import torch
import torch.nn as nn
import torch.nn.functional as F


class Smooth_FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', smoothing=0.0, **kwargs):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # cross entropy
        loss_fct_smooth = LabelSmoothingLoss(smoothing=self.smoothing)
        loss_smooth = loss_fct_smooth(inputs, targets)
        # focal
        loss_fct_focal = FocalLoss()
        loss_focal = loss_fct_focal(inputs, targets)

        return loss_smooth * 0.75 + loss_focal * 0.25


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class Cross_FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # cross entropy
        loss_fct_cross = torch.nn.CrossEntropyLoss()
        loss_cross = loss_fct_cross(inputs, targets)
        # focal
        loss_fct_focal = FocalLoss()
        loss_focal = loss_fct_focal(inputs, targets)

        return loss_cross * 0.75 + loss_focal * 0.25

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        # https://wikidocs.net/60572
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

