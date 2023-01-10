import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs,dim=1)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()
    def forward(self, input, target):
        loss = 0
        # print(type(input))
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.loss(img, label)
            loss += l * w
        return loss 