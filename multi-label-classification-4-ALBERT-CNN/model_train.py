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
from keras.layers import Lambda, concatenate
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalMaxPooling1D
from att import Attention
from keras.layers import Conv1D, Dropout
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

with open("model/rinse/cnn/event_type.json", "w", encoding="utf-8") as h:
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
# print("y_train：", y_train)
# print("y_test:", y_test)
print(y_train.shape)
print(y_test.shape)

# 利用ALBERT对x值（文本）进行编码
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)
print("begin encoding")
f = lambda text: bert_model.encode([text])["encodes"][0]

from keras_bert import Tokenizer, load_vocabulary

# 加载bert字典，构造分词器。
token_dict = load_vocabulary("./albert_zh/albert_tiny_google_zh_489k/vocab.txt")
tokenizer = Tokenizer(token_dict)
# x_train = []
# x_test = []


# 对文本编码
def encoding_text(content_list):
    text = []
    for lines in content_list:
        genres = lines.split(" ", maxsplit=1)[1]
        text.append(genres)
    token_ids = []
    segment_ids = []
    for line in tqdm(text):
        token_id, segment_id = tokenizer.encode(first=line, max_len=512)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    encoding_res = [np.array(token_ids), np.array(segment_ids)]
    return encoding_res


# process_bar = tqdm(train_content)
#
# for ch, line in zip(process_bar, train_content):
#     movie_intro = line.split(" ", maxsplit=1)[1]
#     x_train.append(f(movie_intro))
#
# process_bar = tqdm(test_content)
#
# for ch, line in zip(process_bar, test_content):
#     movie_intro = line.split(" ", maxsplit=1)[1]
#     x_test.append(f(movie_intro))
#

# print("train_content-------->", len(train_content))
x_train = encoding_text(train_content)
x_test = encoding_text(test_content)

# x_train = np.array(x_train)
# x_test = np.array(x_test)
# print("x_train：", x_train)
# print("x_test:", x_test)
print("end encoding")
print(np.array(x_train).shape)


def textcnn(inputs):
    # 选用3、4、5三个卷积核进行特征提取，最后拼接后输出用于分类。
    kernel_size = [3, 4, 5]
    cnn_features = []
    for size in kernel_size:
        cnn = Conv1D(filters=256, kernel_size=size)(
            inputs
        )  # shape=[batch_size,maxlen-2,256]
        cnn = GlobalMaxPooling1D()(cnn)  # shape=[batch_size,256]
        cnn_features.append(cnn)
    # 对kernel_size=3、4、5时提取的特征进行拼接
    output = concatenate(cnn_features, axis=-1)  # [batch_size,256*3]
    # 返回textcnn提取的特征结果
    return output


# 深度学习模型
# 模型结构：ALBERT + CNN  + FC
# CNN层
# embedder = Embedding(len(x_train) + 1, 300)
# embed = embedder(inputs)
# conv_pools = []
# filters = [2, 3, 4]
# for filter in filters:
#     conv = Conv1D(
#         256,
#         kernel_size=filter,
#         padding="valid",
#         activation="relu",
#         kernel_initializer="normal",
#     )(inputs)
#     pooled = MaxPool1D(
#         pool_size=(512 - filter + 1),
#         strides=1,
#         padding="valid",
#     )(conv)
#     conv_pools.append(inputs)

# inputs = Input(
#     shape=(
#         512,
#         312,
#     ),
#     name="input",
# )
# from unit.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification,load_tf_weights_in_albert

# config = AlbertConfig.from_json_file("./albert_zh/albert_tiny/config.json")
# albert = AlbertModel.from_pretrained(
#     "albert_zh/albert_tiny/albert_model.bin", config=config
# )
from bert4keras.models import build_transformer_model

albert = build_transformer_model(
    "./albert_zh/albert_tiny_google_zh_489k/albert_config.json",
    "./albert_zh/albert_tiny_google_zh_489k/albert_model.ckpt",
    model="albert",
    return_keras_model=False,
)

# 取出[cls]，可以直接用于分类，也可以与其它网络的输出拼接。
cls_features = Lambda(lambda x: x[:, 0], name="cls")(
    albert.model.output
)  # shape=[batch_size,768]
# 去除第一个[cls]和最后一个[sep]，得到输入句子的embedding，用作textcnn的输入。
word_embedding = Lambda(lambda x: x[:, 1:-1], name="word_embedding")(
    albert.model.output
)  # shape=[batch_size,maxlen-2,768]
cnn_features = textcnn(word_embedding)
con = concatenate([cls_features, cnn_features], axis=-1)
drop = Dropout(rate=0.5)(con)
den = Dense(256, activation="relu")(drop)
# # attention = Attention(32, name="attention")(drop2)
num_class = len(mlb.classes_)
output = Dense(num_class, activation="sigmoid", name="dense")(den)
model = Model(albert.model.input, output)


# 模型可视化
from keras.utils import plot_model

plot_model(model, to_file="image/cnn/multi-label-model.png", show_shapes=True)

# binary_crossentropy 常见的二分类损失函数，适用于每个样本只能属于一个类别的情况。
# 它将多标签分类问题转化为多个独立的二分类问题，对于每个类别使用一个二分类器来预测该类别的概率。
# 该损失函数使用的是交叉熵的计算方式，可以有效地度量预测结果与真实标签之间的差异。
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10
)
# 保存为\(^o^)/~ h5 结构模型
model.save("./model/rinse/cnn/event_type.h5")

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
plt.savefig("./image/cnn/loss_acc.png")
