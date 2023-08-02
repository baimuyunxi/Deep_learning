#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-label-classification-4-ALBERT-CNN 
@File    ：loss_function.py
@IDE     ：PyCharm 
@Author  ：baimuyunxi
@Date    ：2023/7/26 17:19 
"""
import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss的设计目标是解决在类别不平衡情况下，模型容易忽视少数类别的问题。
    它通过引入一个调节因子来减小易分类的样本的权重，从而使模型更加关注困难样本（即少数类别）。
    这个调节因子由一个可调参数gamma控制，gamma越大，模型对于少数类别的关注度就越高。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class PriorMultiLabelSoftMarginLoss(keras.losses.Loss):
    """
    是一种专门针对多标签分类问题的损失函数。
    它考虑了标签之间的相关性，并且允许每个样本属于多个类别。
    该损失函数在计算损失时，使用了标签之间的相关性信息，以及每个类别的先验概率。
    这种损失函数在处理多标签分类问题时，可以更好地捕捉标签之间的相关性，提高模型的性能。
    """

    def __init__(self, prior=None, reduction=tf.losses.Reduction.NONE, eps=1e-6, tau=1.0):
        super(PriorMultiLabelSoftMarginLoss, self).__init__(reduction=reduction)
        self.prior = prior
        self.eps = eps
        self.tau = tau

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        if self.prior is None:
            num_labels = tf.shape(y_pred)[-1]
            prior = tf.fill([num_labels], 1 / tf.cast(num_labels, tf.float32))
        else:
            prior = tf.constant(self.prior, dtype=tf.float32)

        log_prior = tf.log(prior + self.eps)
        y_pred = y_pred + self.tau * log_prior

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return loss


class CustomBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.losses.Reduction.NONE):
        super(CustomBinaryCrossentropy, self).__init__(reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            # Add a small value to avoid log(0)
            y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
            y_pred = tf.math.log(y_pred / (1 - y_pred))

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return loss