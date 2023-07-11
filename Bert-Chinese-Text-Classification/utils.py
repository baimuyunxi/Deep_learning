# coding: UTF-8
import torch
from tqdm import tqdm
import time
import re
import pandas as pd
import pickle as pkl
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    # ## 读取标签
    label_list = pkl.load(open('./datas/data/labels.pkl', 'rb'))
    print(f"标签个数======== {len(label_list)}")

    def convert_to_one_hot(Y, C):
        list = [[0 for i in C] for j in Y]

        for i, a in enumerate(Y):
            for b in a:
                if b in C:
                    list[i][C.index(b)] = 1
                else:
                    list[i][len(C) - 1] = 1
        return list

    def load_dataset(path, pad_size=200):
        df = pd.read_table(path)
        data = df['text']

        sentences = data

        labels = []
        # 把标签读成数组
        for ls in df['label']:
            labels.append(re.compile(r"'(.*?)'").findall(ls))
        # 把数组转成独热
        labels_id = convert_to_one_hot(labels, label_list)
        contents = []
        count = 0

        for i, content in tqdm(enumerate(sentences)):
            label = labels_id[i]
            encoded_dict = config.tokenizer.encode_plus(
                content,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=pad_size,  # 填充 & 截断长度
                pad_to_max_length=True,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt'  # 返回 pytorch tensors 格式的数据
            )
            token = config.tokenizer.tokenize(content)
            seq_len = len(token)
            count += seq_len
            contents.append((torch.squeeze(encoded_dict['input_ids'],0), label, seq_len, torch.squeeze(encoded_dict['attention_mask'],0)))
        print(f"数据集地址========{path}")
        print(f"数据集总词数========{count}")
        print(f"数据集文本数========{len(sentences)}")
        print(f"数据集文本平均词数========{count / len(sentences)}")
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0].detach().numpy() for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3].detach().numpy() for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
