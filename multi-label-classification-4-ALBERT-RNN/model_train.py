# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-04-03 18:12

import json
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from att import Attention
from keras.layers import GRU, Bidirectional
from tqdm import tqdm
import matplotlib.pyplot as plt
from albert_zh.extract_feature import BertVector

with open("./data/rinse4_5/data_train.txt", "r", encoding="utf-8") as f:
    train_content = [_.strip() for _ in f.readlines() if _ is not None]

with open("./data/rinse4_5/data_dev.txt", "r", encoding="utf-8") as f:
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

with open("model/rinse/rgu4_5/event_type.json", "w", encoding="utf-8") as h:
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
# conv1 = Conv1D(128, 3, activation="relu")(inputs)
# pool1 = MaxPooling1D(pool_size=2)(conv1)
gru = Bidirectional(GRU(128, dropout=0.2, return_sequences=True), name="bi-gru")(inputs)
attention = Attention(32, name="attention")(gru)
num_class = len(mlb.classes_)
output = Dense(num_class, activation="relu", name="dense")(attention)
model = Model(inputs, output)

# CNN层
# cnn = Conv1D(filters=64, kernel_size=3, activation='relu', name='cnn')(inputs)
# # 池化层
# pooling = MaxPooling1D(pool_size=2, name='pooling')(cnn)
# # 双向LSTM层
# lstm = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True), name="bi-lstm")(pooling)
# # Attention层
# attention = Attention(32, name="attention")(lstm)
# num_class = len(mlb.classes_)
#
# # 全连接层
# output = Dense(num_class, activation="relu", name="dense")(attention)
#
# model = Model(inputs, output)


# 模型可视化
from keras.utils import plot_model

plot_model(model, to_file="./image/rgu4_5/multi-label-model.png", show_shapes=True)


# binary_crossentropy 常见的二分类损失函数，适用于每个样本只能属于一个类别的情况。
# 它将多标签分类问题转化为多个独立的二分类问题，对于每个类别使用一个二分类器来预测该类别的概率。
# 该损失函数使用的是交叉熵的计算方式，可以有效地度量预测结果与真实标签之间的差异。
model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

history = model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=10
)
# 保存为\(^o^)/~ h5 结构模型
model.save("./model/rinse/rgu4_5/event_type.h5")

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
plt.savefig("./image/rgu4_5/loss_acc.png")
