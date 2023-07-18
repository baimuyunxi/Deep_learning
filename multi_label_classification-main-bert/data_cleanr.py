#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi_label_classification-main
@File    ：data_cleanr.py
@IDE     ：PyCharm
@Author  ：baimuyunxi
@Date    ：2023/7/17 20:04
"""

import pandas as pd
import json
import os
import random

l_train = []
l_val = []
l_test = []


# 读取文件中的内容，并将其打乱写入列表FileNameList
def ReadFileDatas(original_filename):
    file = open(original_filename, "r+")
    FileNameList = file.readlines()
    random.shuffle(FileNameList)
    file.close()
    print("数据集总量：", len(FileNameList))
    return FileNameList


# 将数据集随机划分
def TrainValTestFile(FileNameList):
    i = 0
    j = len(FileNameList)
    for line in FileNameList:
        if i < (j * 0.6):
            i += 1
            l_train.append(line)
        elif i < (j * 0.8):
            i += 1
            l_val.append(line)
        else:
            i += 1
            l_test.append(line)
    print("总数量:%d,此时创建train,val,test数据集" % i)
    return l_train, l_val, l_test


# 将获取到的各个数据集的包含的文件名写入txt中
def WriteDatasToFile(listInfo, new_filename):
    file_handle = open(new_filename, "w", encoding="utf-8")
    for str_Result in listInfo:
        json_str = json.dumps(str_Result, ensure_ascii=False)
        file_handle.write(json_str + "\n")
    file_handle.close()
    print("写入 %s 文件成功." % new_filename)


# 调用函数
if __name__ == "__main__":
    csv_file = "./data/file_0.csv"
    # txt_file = "./data/new_data.json"
    df = pd.read_csv(csv_file)
    df = df.dropna()
    # listFileInfo = len(df)  # 获取行数
    # print(listFileInfo)
    df["1"] = df["1"].str.split("-")
    df_data = df[["0", "1"]]
    df_data.rename(columns={"0": "text", "1": "label"}, inplace=True)
    data = df_data.to_dict(orient="records")

    l_train, l_val, l_test = TrainValTestFile(data)
    WriteDatasToFile(data, "./data/data_new.json")
    WriteDatasToFile(l_train, "./data/train.json")
    WriteDatasToFile(l_val, "./data/dev.json")
    WriteDatasToFile(l_test, "./data/test.json")
