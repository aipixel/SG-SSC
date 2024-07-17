import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_class_weights(dataloader, num_classes, dev):
    # get class frequencies
    freq = np.zeros((num_classes,))
    for _, labels in dataloader:
        for c in range(num_classes):
            freq[c] += torch.sum(labels == c)
    class_probs = freq / np.sum(freq)

    class_weights = np.append((1/class_probs)/np.sum(1/class_probs),0)

    print(class_weights)
    print(sum(class_weights))
    return torch.from_numpy(class_weights.astype(np.float32)).to(dev)


class CustomCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=40) -> None:
        super(CustomCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #print(input.shape)
        #print(target.shape)

        input = input.view(input.size(0), input.size(1), -1).contiguous()  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C

        target = target.view(target.size(0), -1).contiguous()  # N,H,W => N,H*W

        #print(input.shape)
        #print(target.shape)

        idx = target < self.num_classes

        input = input[idx]
        target = target[idx]

        return super(CustomCrossEntropy, self).forward(input, target)


class MscCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', num_classes=11):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, preds, target):
        loss = 0
        for item in preds:
            #print('item.shape:',item.shape)
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='nearest').long()
           
            item = item.view(item.size(0), item.size(1), -1).contiguous()  # N,C,H,W => N,C,H*W
            item = item.transpose(1, 2)  # N,C,H*W => N,H*W,C

            item_target = item_target.view(item_target.size(0), -1).contiguous()  # N,H,W => N,H*W
            
            idx = item_target < self.num_classes
            item = item[idx]
            item_target = item_target[idx]
           
            loss += F.cross_entropy(item, item_target, weight=self.weight, reduction=self.reduction)

        return loss / len(preds)


class SSCCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=2) -> None:
        super(SSCCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        #print(input.shape)
        #print(target.shape)

        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C

        target = target.view(target.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        #print(input.shape)
        #print(target.shape)

        epsilon = 1e-10

        occluded = torch.logical_and(~torch.abs(weights).eq(1.0), ~weights.eq(0))

        occupied = torch.abs(weights).eq(1.0)

        #ratio = (4. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)
        ratio = (2. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)
        #ratio = (1. * torch.sum(occupied)) / (torch.sum(occluded) + epsilon)

        rand = torch.rand(size=list(target.size())).to(weights.device)

        idx = torch.logical_or(occupied, torch.logical_and(occluded, rand <= ratio))

        input = input[idx]
        target = target[idx]

        return super(SSCCrossEntropy, self).forward(input, target.type(torch.LongTensor).to(weights.device))


class BCELoss(nn.Module):
    def __init__(self, weight: Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    def forward(self, logits, labels, weights):
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        logits = logits.view(labels.size(0), -1)
        labels = labels.view(labels.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        idx = weights != 0.0

        logits = logits[idx]
        labels = labels[idx].float()
        #print(logits.shape)
        #print(labels.shape)
        
        loss = self.bce(logits, labels)
       
        return loss


class WeightedSSCCrossEntropy(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Tensor = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', num_classes=12) -> None:
        super(WeightedSSCCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.num_classes = num_classes

    def forward(self, logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
        #print(input.shape)
        #print(target.shape)
        logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
        logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C

        labels = labels.view(labels.size(0), -1)  # N,H,W => N,H*W
        weights = weights.view(weights.size(0), -1)  # N,H,W => N,H*W

        idx = weights != 0.0

        logits = logits[idx]
        labels = labels[idx]

        return super(WeightedSSCCrossEntropy, self).forward(logits, labels.type(torch.LongTensor).to(weights.device))


def weighted_categorical_crossentropy(y_pred, y_true, weights):

    epsilon = 1e-10

    occluded = torch.logical_and(~torch.abs(weights).eq(1.0), ~weights.eq(0))

    occupied = torch.abs(weights).eq(1.0)

    ratio = (2. * torch.sum(occupied))/ (torch.sum(occluded) + epsilon)

    rand = torch.rand(size=list(y_true.size())).to(weights.device)

    print(ratio)

    w = torch.logical_or(occupied, torch.logical_and(occluded, rand<=ratio)).float()

    norm_w = (w/(torch.mean(w)+ epsilon)).reshape((w.size(0),1,w.size(1),w.size(2),w.size(3) )).repeat(1,12,1,1,1)


    # clip to prevent NaN's and Inf's
    #y_pred = torch.clip(y_pred, epsilon, 1 - epsilon)


    log_y_pred = torch.nn.LogSoftmax(dim=1)(y_pred)
    y_true = torch.movedim(torch.nn.functional.one_hot(y_true.type(torch.LongTensor), num_classes=12),4,1).to(weights.device)


    # calc
    #print(y_true.device)
    #print(log_y_pred.device)
    #print(norm_w.device)
    loss = -torch.mean(y_true * log_y_pred * norm_w)
    return loss
