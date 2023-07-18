# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        # bert-base-chinese 请提前下载好放入./bert-base-chinese，可以避免每次运行下载一次，下载地址见：./bert-base-chinese/README.md
        # 修改模型张量， 默认为512
        self.bert = BertModel.from_pretrained(
            "./bert-base-chinese/",
            max_position_embeddings=1024,
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls = self.drop(outputs[1])
        out = F.sigmoid(self.fc(cls))
        return out
