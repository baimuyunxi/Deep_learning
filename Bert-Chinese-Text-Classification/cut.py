# -*- encoding: utf-8 -*-
'''
@File    :   cut.py
@Time    :   2023/07/11 15:09:41
@Author  :   Yunxi 
@Version :   1.0
@Contact :   baimuyunxi@163.com
@License :   (C)Copyright 2017-2023, Liugroup-NLPR-CASIA
'''

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data_bak = pd.read_table('./data/class1.txt')
data = data_bak.dropna()
print(data.isna().sum())
# 假设CSV文件中的文本数据所在的列名为 'text'，标签所在的列名为 'label'
texts = data['0']
labels = data['1']

# 将数据集拆分为训练集、验证集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# 创建新的数据框保存拆分后的数据集
train_data = pd.DataFrame({'text': train_texts, 'label': train_labels})
val_data = pd.DataFrame({'text': val_texts, 'label': val_labels})
test_data = pd.DataFrame({'text': test_texts, 'label': test_labels})

# 保存拆分后的数据集为txt文件
train_data.to_csv('./data/word_frequency/train.txt', sep='\t', index=False)
val_data.to_csv('./data/word_frequency/dev.txt', sep='\t', index=False)
test_data.to_csv('./data/word_frequency/test.txt', sep='\t', index=False)

# 使用 pickle 进行压缩存储
with open('./data/word_frequency/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

# 从存储的文件中读取压缩的标签
with open('./data/word_frequency/labels.pkl', 'rb') as f:
    stored_labels = pickle.load(f)

# 打印读取到的标签
print(stored_labels)
