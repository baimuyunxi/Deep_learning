#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-label-classification-4-ALBERT-RNN
@File    ：model_train.py
@IDE     ：PyCharm
@Author  ：baimuyunxi
@Date    ：2023/7/13 23:05
"""

import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.layers import Input, Dense
from att import Attention
from keras.layers import (
    GRU,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    concatenate,
    Dropout,
    Flatten,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from albert_zh.extract_feature import BertVector

with open("./data/rinse4_5/data_train.txt", "r", encoding="utf-8") as f:
    train_content = [_.strip() for _ in f.readlines() if _ is not None]

with open("./data/rinse4_5/data_test.txt", "r", encoding="utf-8") as f:
    test_content = [_.strip() for _ in f.readlines() if _ is not None]

# 获取训练集合、测试集的事件类型
movie_genres = []

for line in train_content + test_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    movie_genres.append(genres)

# 利用sklearn中的MultiLabelBinarizer进行多标签编码  机器学习中 hot-one 编码
mlb = MultiLabelBinarizer()
mlb.fit(movie_genres)

print("一共有%d种事件类型。" % len(mlb.classes_))

with open("model/rinse/test/event_type.json", "w", encoding="utf-8") as h:
    h.write(json.dumps(mlb.classes_.tolist(), ensure_ascii=False, indent=4))

# 对训练集和测试集的数据进行多标签编码
y_train = []
y_test = []

for line in train_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    y_train.append(mlb.transform([genres])[0])

for line in test_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    y_test.append(mlb.transform([genres])[0])

y_train = np.array(y_train)
y_test = np.array(y_test)

print(y_train.shape)
print(y_test.shape)

# 利用ALBERT对x值（文本）进行编码
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)
print("begin encoding")
f = lambda text: bert_model.encode([text])["encodes"][0]

x_train = []
x_test = []

process_bar = tqdm(train_content)

for ch, line in zip(process_bar, train_content):
    movie_intro = line.split(" ", maxsplit=1)[1]
    x_train.append(f(movie_intro))

process_bar = tqdm(test_content)

for ch, line in zip(process_bar, test_content):
    movie_intro = line.split(" ", maxsplit=1)[1]
    x_test.append(f(movie_intro))

x_train = np.array(x_train)
x_test = np.array(x_test)

print("end encoding")
print(x_train.shape)

# 深度学习模型
# 模型结构：ALBERT + 双向GRU + Attention + FC
inputs = Input(
    shape=(
        512,
        312,
    ),
    name="input",
)
gru = Bidirectional(GRU(128, dropout=0.2, return_sequences=True), name="bi-gru")(inputs)
# CNN层
# 卷积层和池化层，设置卷积核大小分别为3,4,5
# cnn1 = Conv1D(256, 3, padding="same", strides=1, activation="relu")(inputs)
# cnn1 = MaxPooling1D(pool_size=48)(cnn1)
# cnn2 = Conv1D(256, 4, padding="same", strides=1, activation="relu")(inputs)
# cnn2 = MaxPooling1D(pool_size=47)(cnn2)
# cnn3 = Conv1D(256, 5, padding="same", strides=1, activation="relu")(inputs)
# cnn3 = MaxPooling1D(pool_size=46)(cnn3)
# # 合并三个模型的输出向量
# cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
# flat = Flatten()(cnn)
# drop = Dropout(0.2)(flat)  # 在池化层到全连接层之前可以加上dropout防止过拟合

attention = Attention(32, name="attention")(gru)
num_class = len(mlb.classes_)
output = Dense(num_class, activation="relu", name="dense")(attention)
model = Model(inputs, output)


# 模型可视化
from keras.utils import plot_model

plot_model(model, to_file="./image/test/multi-label-model.png", show_shapes=True)

# binary_crossentropy 常见的二分类损失函数，适用于每个样本只能属于一个类别的情况。
# 它将多标签分类问题转化为多个独立的二分类问题，对于每个类别使用一个二分类器来预测该类别的概率。
# 该损失函数使用的是交叉熵的计算方式，可以有效地度量预测结果与真实标签之间的差异。
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=10
)
# 保存为\(^o^)/~ h5 结构模型
model.save("./model/rinse/test/event_type.h5")

# 训练结果可视化
# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history["loss"])
plt.plot(range(epochs), history.history["loss"], label="loss")
plt.plot(range(epochs), history.history["val_loss"], label="val_loss")
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history["accuracy"])
plt.plot(range(epochs), history.history["accuracy"], label="acc")
plt.plot(range(epochs), history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.savefig("./image/test/loss_acc.png")
