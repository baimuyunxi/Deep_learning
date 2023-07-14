# coding: UTF-8
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import time
from utils import get_time_dif, Logger
from transformers import AdamW, get_linear_schedule_with_warmup

logger = Logger(os.path.join("datas/log", "log.txt"))
# 定义损失函数
Loss = nn.BCEWithLogitsLoss()


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


# 计算多标签准确率、精确率、hm
def APH(y_true, y_pred):
    return (
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred, average='samples'),
        metrics.hamming_loss(y_true, y_pred),
        # metrics.precision_recall_fscore_support(y_true, y_pred),
    )


# 预测多标签的输出，把概率值转化为独热数组
def Predict(outputs, alpha=0.4):
    predic = torch.relu(outputs)
    zero = torch.zeros_like(predic)
    topk = torch.topk(predic, k=2, dim=1, largest=True)[1]
    for i, x in enumerate(topk):
        for y in x:
            if predic[i][y] > alpha:
                zero[i][y] = 1
    return zero.cpu()


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
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_iter) * config.num_epochs
    )
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    if is_write:
        # 可视化工具。它用于记录训练过程中的各种指标、损失和参数等信息
        writer = SummaryWriter(
            log_dir="{0}/{1}__{2}__{3}__{4}".format(
                config.log_path,
                config.batch_size,
                config.pad_size,
                config.learning_rate,
                time.strftime('%m-%d_%H.%M', time.localtime()),
            )
        )
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            # print('---------train----------')
            # print('裁剪前:', outputs.shape, '<----->', labels.shape)
            # 对输出张量进行裁剪
            labels = labels[:, : outputs.shape[1]]
            # print('裁剪后:', outputs.shape, '<----->', labels.shape)
            # print('type:', type(outputs), '<----->', type(labels))
            # 计算损失 (输入，目标）
            loss = Loss(outputs, labels.float())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels
                predic = Predict(outputs)
                # train_oe = OneError(outputs, true)
                train_acc, train_pre, train_hl = APH(true.data.cpu().numpy(), predic.data.cpu().numpy())

                dev_acc, dev_pre, dev_hl, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = (
                    'Iter: {0:>6}, Train=== Loss: {1:>6.2}, Acc: {2:>6.2%}, Pre: {3:>6.2%}, HL: {4:>5.2}, Val=== Loss: {5:>5.2}, Acc: {6:>6.2%}, Pre: {7:>6.2%}, HL: {8:>5.2}, '
                    'OE: Null, Time: {9} {10} '
                )
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        train_pre,
                        train_hl,
                        # train_oe,
                        dev_loss,
                        dev_acc,
                        dev_pre,
                        dev_hl,
                        # dev_oe,
                        time_dif,
                        improve,
                    )
                )
                if is_write:
                    writer.add_scalar('loss/train', loss.item(), total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("pre/train", train_pre, total_batch)
                    # writer.add_scalar("oe/train", train_oe, total_batch)
                    writer.add_scalar("hamming loss/train", train_hl, total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    writer.add_scalar("pre/dev", dev_pre, total_batch)
                    # writer.add_scalar("oe/dev", dev_oe, total_batch)
                    writer.add_scalar("hamming loss/dev", dev_hl, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        scheduler.step()  # 学习率衰减
        if flag:
            break
    if is_write:
        writer.close()
    return test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_pre, test_rec, test_hl, test_loss = evaluate(
        config, model, test_iter, test=True
    )
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Pre: {2:>6.2%}, Test HL: {3:>5.2}, Test OE: {4:>6.2%}'
    print(msg.format(test_loss, test_acc, test_pre, test_rec, test_hl))
    print("Precision, Recall and F1-Score...")
    # print(test_report)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return test_loss, test_acc, test_pre, test_rec, test_hl


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # oe = OneError(outputs.data.cpu(), labels.data.cpu())
            # print('---------evaluate----------')
            # print('裁剪前:', outputs.shape, '<----->', labels.shape)
            # 对输出张量进行裁剪
            labels = labels[:, : outputs.shape[1]]
            # print('裁剪后:', outputs.shape, '<----->', labels.shape)
            # print('type:', type(outputs), '<----->', type(labels))
            loss = Loss(outputs, labels.float())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = Predict(outputs.data)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic.numpy())

    labels_all = labels_all.reshape(-1, config.num_classes)
    predict_all = predict_all.reshape(-1, config.num_classes)
    acc, pre, hl = APH(labels_all, predict_all)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=3
        )
        return acc, pre, hl, loss_total / len(data_iter), report
    return acc, pre, hl, loss_total / len(data_iter)
