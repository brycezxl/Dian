import torch
from torch import nn
import torch.nn.functional as f


class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = f.softmax(pred, dim=1)
        # one_hot_label = f.one_hot(label, pred.size(1)).float()
        one_hot_label = torch.zeros((label.size(0), 3)).cuda()
        for i in range(label.size(0)):
            one_hot_label[i, label[i]] = 1
        smoothed_one_hot_label = (1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = torch.sum(loss, dim=1)
        loss = loss.mean()
        return loss
