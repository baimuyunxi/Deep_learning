# coding: UTF-8
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, Logger
from transformers import AdamW, get_linear_schedule_with_warmup

logger = Logger(os.path.join("datas/log", "log.txt"))


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, is_write):
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("dropout", config.dropout)
    logger.log("Max Sequence Length:", config.pad_size)
    logger.log()

    start_time = time.time()
    model.train()

    # 普通算法
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # bert算法
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    # BertAdam implements weight decay fix,
    # BertAdam doesn't compensate for bias as in the regular Adam optimizer.
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    total_step = len(train_iter) * config.num_epochs
    num_warmup_steps = round(total_step * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_step
    )

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logger.log(
                    msg.format(
                        total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve
                    )
                )
                print(
                    msg.format(
                        total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve
                    )
                )
                if is_write:
                    writer.add_scalar('loss/train', loss.item(), total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("pre/train", train_pre, total_batch)
                    writer.add_scalar("oe/train", train_oe, total_batch)
                    writer.add_scalar("hamming loss/train", train_hl, total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    writer.add_scalar("pre/dev", dev_pre, total_batch)
                    writer.add_scalar("oe/dev", dev_oe, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logger.log(msg.format(test_loss, test_acc))
    print(msg.format(test_loss, test_acc))
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Pre: {2:>6.2%}, Test HL: {3:>5.2}, Test OE: {4:>6.2%}'
    # logger.log(msg.format(test_loss, test_acc, test_pre, test_rec, test_hl))
    # print(msg.format(test_loss, test_acc, test_pre, test_rec, test_hl))

    print("Precision, Recall and F1-Score...")
    logger.log(test_report)
    # print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # return test_loss, test_acc, test_pre, test_rec, test_hl


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # oe = OneError(outputs.data.cpu(), labels.data.cpu())
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=4
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


# 计算多标签准确率、精确率、hm
def APH(y_true, y_pred):
    return (
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred, average='samples'),
        metrics.hamming_loss(y_true, y_pred),
    )


# 预测多标签的输出，把概率值转化为独热数组
def Predict(outputs, alpha=0.4):
    predic = torch.sigmoid(outputs)
    zero = torch.zeros_like(predic)
    topk = torch.topk(predic, k=2, dim=1, largest=True)[1]
    for i, x in enumerate(topk):
        for y in x:
            if predic[i][y] > alpha:
                zero[i][y] = 1
    return zero.cpu()
