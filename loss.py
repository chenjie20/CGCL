import torch
import torch.nn.functional as F


def binary_cross_entropy_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss
