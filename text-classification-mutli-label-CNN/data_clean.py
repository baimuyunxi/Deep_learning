#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：text-classification-mutli-label-master 
@File    ：data_clean.py
@IDE     ：PyCharm 
@Author  ：baimuyunxi
@Date    ：2023/7/12 22:20 
"""
import random
import pandas as pd
import jieba

# jieba 线程加速
# jieba.enable_parallel()

L_train = []
L_val = []
L_test = []

jieba.load_userdict("./datas/jb_word/dict_word.txt")
with open("./datas/jb_word/cn_stopwords_set.txt", "r", encoding="utf-8") as f:
    stopword = [line.strip() for line in f]


# 停用词列表
def cut_word(words):
    # 使用jieba进行分词，并排除停用词
    seg_list = [word for word in jieba.cut(str(words)) if word not in stopword]
    # 按照空格拼接分词结果
    result = " ".join(seg_list)
    return result


def cut_label(words):
    result = " __label__".join(words)
    result = "__label__" + result  # 添加拼接的开头
    return result


# 读取文件中的内容，并写入列表FileNameList
def ReadFileDatas(original_filename):
    FileNameList = []
    file = open(original_filename, "r+", encoding="utf-8")
    for line in file:
        FileNameList.append(line)  # 写入文件内容到列表中去
    print("数据集总量：", len(FileNameList))
    file.close()
    return FileNameList


# 将获取的列表中的内容转为 str ，再写入到txt文件中去
# listInfo为 ReadFileDatas 的列表
def WriteDatasToFile(listInfo, new_filename):
    file_handle = open(new_filename, mode="a", encoding="utf-8")
    for idx in range(len(listInfo)):
        str = listInfo[idx]  # 列表指针
        str_Result = str
        file_handle.write(str_Result)
    file_handle.close()
    print("写入 %s 文件成功." % new_filename)


"""
将划分数据集用函数表示
划分数据集（train, val, test）的区间，（new.txt） 为随机打乱好的文件数据集
数据集列表集合
打开文件引用上一函数保存的文件
"""


def TrainValTestFile(new_filename):
    i = 0  # counter
    j = 9352  # all lines
    file_divide = open(new_filename, "r", encoding="utf-8")
    lines = file_divide.readlines()
    for line in lines:
        if i < (j * 0.6):
            i += 1
            L_train.append(line)
        elif i < (j * 0.8):
            i += 1
            L_val.append(line)
        elif i < j:
            i += 1
            L_test.append(line)
    print("总数据量：%d , 此时创建train, val, test数据集" % i)
    return L_train, L_val, L_test


# 保存数据集（train, val, test）
def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表
    file = open(filename, "a", encoding="utf-8")
    for i in range(len(data)):
        s = str(data[i]).replace("[", "").replace("]", "")  # 去除[],这两行按数据不同，可以选择
        # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存数据集（路径）成功：%s" % filename)


# 数据清洗
def dataParse(data):
    data_text = data["0"]
    data_label = data["1"]
    print(data_text, data_label)
    # print(data_label[1], type(data_label[1]))
    data_label = data_label.map(eval)
    # print(data_label[1], type(data_label[1]))
    # 合并元素
    merged_data = data_text.map(cut_word) + " " + data_label.map(cut_label)
    # # print(merged_data)
    # output_file = f"./datas/data/data_{names}.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write("\n".join(merged_data))
    output_file = "./datas/data/data.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_data))

    listFileInfo = ReadFileDatas(output_file)  # 读取文件
    random.shuffle(listFileInfo)  # 打乱顺序
    WriteDatasToFile(listFileInfo, "./datas/new_data.txt")  # 保存新的文件

    # 划分数据集并保存
    TrainValTestFile("./datas/new_data.txt")
    text_save("./datas/data/data_train.txt", L_train)
    text_save("./datas/data/data_val.txt", L_val)
    text_save("./datas/data/data_test.txt", L_test)


def cut_data(datas):
    data_bak = pd.read_csv(datas)
    data = data_bak.dropna()
    # 保存拆分后的数据集为txt文件
    dataParse(data)


if __name__ == "__main__":
    cut_data("./datas/data/class1.csv")
