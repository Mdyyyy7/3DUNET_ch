import numpy as np
import torch
import torch.nn.functional as F




    

def manual_crossentropy(target, output, from_logits=False, dim=-1):
    if not from_logits:
        output = output / torch.sum(output, dim=dim, keepdim=True)
        eps = torch.finfo(output.dtype).eps
        output = torch.clamp(output, eps, 1. - eps)
        return -torch.sum(target * torch.log(output), dim=dim)
    else:
        return F.cross_entropy(output, target.argmax(dim=dim), reduction='none')
        
# 求平均损失
def crossentropy(dim=-1):
    def loss(y_true, y_pred):
        return torch.mean(manual_crossentropy(y_true, y_pred, dim=dim))
    return loss

def soft_dice(y_true, y_pred):
    smooth = 1.0
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=1)
    numerator = 2. * intersection + smooth
    denominator = torch.sum(y_true ** 2, dim=1) + torch.sum(y_pred ** 2, dim=1) + smooth
    coeff = numerator / denominator
    return 1. - coeff

def weighted_categorical_crossentropy(dim=1, from_logits=False, classes=2):
    def loss(y_true, y_pred):
        L = manual_crossentropy(y_true, y_pred, from_logits=from_logits, dim=dim)
        y_true_p = y_true.argmax(dim=dim)
        total_loss = 0.0
        eps = 1e-7
        for c in range(classes):
            mask = (y_true_p == c).float()
            w = 1.0 / (mask.sum() + eps)
            total_loss += (L * mask * w).sum()
        return total_loss
    return loss


def weighted_categorical_crossentropy_with_fpr(dim=1, from_logits=False, classes=2, threshold=0.5):
    def loss(y_true, y_pred):
        L = manual_crossentropy(y_true, y_pred, from_logits=from_logits, dim=dim)
        y_true_p = y_true.argmax(dim=dim)
        y_pred_probs = y_pred if from_logits else torch.max(y_pred, dim=dim).values
        y_pred_bin = (y_pred >= threshold).float() if from_logits else y_pred.argmax(dim=dim)

        total_loss = 0.0
        eps = 1e-7
        for c in range(classes):
            c_true = (y_true_p == c).float()
            w = 1.0 / (c_true.sum() + eps)
            total_loss += (L * c_true * w).sum()

            c_false_p = ((y_true_p != c) & (y_pred_bin == c)).float()
            gamma = 0.5 + (torch.sum(torch.abs((c_false_p * y_pred_probs) - 0.5)) / (c_false_p.sum() + eps))
            wc = w * gamma
            total_loss += (L * c_false_p * wc).sum()
        return total_loss
    return loss