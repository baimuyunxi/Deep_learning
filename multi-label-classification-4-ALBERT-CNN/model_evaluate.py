#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-label-classification-4-ALBERT-RNN
@File    ：model_evaluate.py
@IDE     ：PyCharm
@Author  ：baimuyunxi
@Date    ：2023/7/13 23:05
"""
# 模型评估脚本,利用hamming_loss作为多标签分类的评估指标，该值越小模型效果越好
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import hamming_loss, classification_report
from att import Attention
from albert_zh.extract_feature import BertVector
import tensorflow as tf

# 模型预测处添加这行,关闭eager模式
tf.compat.v1.disable_eager_execution()

# 加载训练好的模型
model = load_model(
    "model/rinse/cnn/event_type.h5", custom_objects={"Attention": Attention}
)

with open("model/rinse/cnn/event_type.json", "r", encoding="utf-8") as f:
    event_type_list = json.loads(f.read())
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)


# 文本长度处理
def text_between_customer(text_series):
    """
    源码中 length 修改裁剪长度
    """
    length = 510
    text_series = str(text_series).replace(" ", "")
    start_index = text_series.find("客户")
    end_index = text_series.rfind("坐席")

    if len(text_series) <= length or start_index == -1 or end_index == -1:
        return text_series

    # 防止客户开头包含重要信息 & 客户开头标识在文中间位置去了
    if 0 <= start_index <= 4 or start_index >= 18:
        extracted_text = text_series[:end_index].strip()

    # 结尾处理 反正结尾没识别到坐席
    elif end_index < len(text_series) - 50:
        extracted_text = text_series[start_index + len("客户：") :].strip()
    # 正常处理
    else:
        extracted_text = text_series[start_index + len("客户：") : end_index].strip()
    # print('start_index：', start_index, 'end_index', end_index)
    # 限制提取的文本长度的字符
    if len(extracted_text) > length:
        front_len = length * 4 // 5  # 前段取4/5长度
        extracted_text = (
            extracted_text[:front_len] + extracted_text[-(length - front_len) :]
        )
    return extracted_text


# 对单句话进行预测
def predict_single_text(text):
    # 将句子转换成向量
    text_df = text_between_customer(text)
    vec = bert_model.encode([text_df])["encodes"][0]
    x_train = np.array([vec])

    # 模型预测
    predicted = model.predict(x_train)[0]
    indices = sorted(
        [i for i in range(len(predicted)) if predicted[i] > 0.2],
        key=lambda i: predicted[i],
        reverse=True,
    )[:5]
    one_hot = [0] * len(event_type_list)
    for index in indices:
        one_hot[index] = 1

    return one_hot, "|".join([event_type_list[index] for index in indices])


# 模型评估
def evaluate():
    with open("./data/rinse4_5/data_test.txt", "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    true_y_list, pred_y_list = [], []
    true_label_list, pred_label_list = [], []
    common_cnt = 0
    for i in range(len(content)):
        print("predict %d samples" % (i + 1))
        true_label, text = content[i].split(" ", maxsplit=1)
        true_y = [0] * len(event_type_list)
        for i, event_type in enumerate(event_type_list):
            if event_type in true_label:
                true_y[i] = 1

        pred_y, pred_label = predict_single_text(text)
        if set(true_label.split("-")) == set(pred_label.split("-")):
            common_cnt += 1
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)
        true_label_list.append(true_label)
        pred_label_list.append(pred_label)

    # F1值
    print(
        classification_report(
            true_y_list,
            pred_y_list,
            digits=4,
        )
    )
    return (
        true_label_list,
        pred_label_list,
        hamming_loss(true_y_list, pred_y_list),
        common_cnt / len(true_y_list),
    )


# 输出模型评估结果
true_labels, pred_lables, h_loss, accuracy = evaluate()
df = pd.DataFrame({"y_true": true_labels, "y_pred": pred_lables})
df.to_csv("model/rinse/cnn/pred_result.csv", index=False)

print("accuracy: ", accuracy)
print("hamming loss: ", h_loss)
