#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-label-classification-4-ALBERT-CNN 
@File    ：loss_function.py
@IDE     ：PyCharm 
@Author  ：baimuyunxi
@Date    ：2023/7/26 17:19 
"""
import numpy as np
from torch import nn
import torch
import tensorflow as tf


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        """FocalLoss
        聚焦损失, 不确定的情况下alpha==0.5效果可能会好一点
        url: https://github.com/CoinCheung/pytorch-loss
        Usage is same as nn.BCEWithLogits:
          >>> loss = criteria(logits, lbs)
        """
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(labels - probs).pow(self.gamma).neg()
        log_0_probs = torch.where(
            logits >= 0,
            -logits + nn.functional.softplus(logits, -1, 50),
            -nn.functional.softplus(logits, 1, 50),
        )
        log_1_probs = torch.where(
            logits >= 0,
            nn.functional.softplus(logits, -1, 50),
            logits - nn.functional.softplus(logits, 1, 50),
        )
        loss = (
            labels * self.alpha * log_1_probs
            + (1.0 - labels) * (1.0 - self.alpha) * log_0_probs
        )
        loss = loss * coeff
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


class PriorMultiLabelSoftMarginLoss(nn.Module):
    def __init__(
        self, prior=None, num_labels=None, reduction="mean", eps=1e-6, tau=1.0
    ):
        """PriorCrossEntropy
        categorical-crossentropy-with-prior
        urls: [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
        args:
            prior: List<float>, prior of label, 先验知识.  eg. [0.6, 0.2, 0.1, 0.1]
            num_labels: int, num of labels, 类别数.  eg. 10
            reduction: str, Specifies the reduction to apply to the output, 输出形式.
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 1e-9
            tau: float, weight of prior in loss, 先验知识的权重, eg. ``1.0``
        returns:
            Tensor of loss.
        examples:
        >>> loss = PriorCrossEntropy(prior)(logits, label)
        """
        super(PriorMultiLabelSoftMarginLoss, self).__init__()
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss(reduction=reduction)
        if prior is None:
            prior = np.array(
                [1 / num_labels for _ in range(num_labels)]
            )  # 如果不存在就设置为num
        if type(prior) == list:
            prior = np.array(prior)
        self.log_prior = torch.tensor(np.log(prior + eps)).unsqueeze(0)
        self.eps = eps
        self.tau = tau

    def forward(self, logits, labels):
        # 使用与输入label相同的device
        logits = logits + self.tau * self.log_prior.to(labels.device)
        loss = self.loss_mlsm(logits, labels)
        return loss
