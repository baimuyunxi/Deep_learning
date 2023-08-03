#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-label-classification-4-event-type-master
@File    ：model_predict.py
@IDE     ：PyCharm
@Author  ：baimuyunxi
@Date    ：2023/7/13 23:05
"""

import json
import pickle

import numpy as np
from keras.models import load_model

from att import Attention
from albert_zh.extract_feature import BertVector
import tensorflow as tf

# 模型预测处添加这行,关闭eager模式
tf.compat.v1.disable_eager_execution()

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
        mid_maxlen = length // 2 - 1
        extracted_text = extracted_text[:mid_maxlen] + extracted_text[-mid_maxlen:]
    return extracted_text


load_model = load_model(
    "model/rinse/cnn/event_type.h5", custom_objects={"Attention": Attention}
)


# 预测语句
text = "坐席：很高，兴为您服务您好。客户：唉喂你好，我就是我这个号码不知道怎么回事。叫做西门叫科技，有些地方能打通电话，然后回到邵阳打电话打不通了。那个流量还是能用，然后显示也是4G信号？坐席：直接打电话打不了是吧。客户：对对对，我现在打打，你打了10个电话才打通，坐席：地址具体的手机位置。客户：手手机位置嗯邵阳嗯邵阳市嗯，还要详细的。坐席：对啊城市里面提供的小区、乡、镇提供的乡、镇、村？客户：那噢我办的这个卡是是吧。坐席：不，是办的地方，我是说您现在的使用地址客户：我现在的位置啊上就是，坐席：上市然后呢。客户：嗯我也不晓得。这个。坐席：2块9，您那边稍微的走动一下，您声音断断续续，我听不清完整的一个句话，两个词语之后就没声音了。客户：那个不是很好？坐席：那您如果是这如果是这种情，况您方便的话可以选择用其他的更信号好的电话，打然后报这个号码的故障就行了，因为这样子方便沟通，您现在说两个字后面就听不清了，您现在是邵阳的什么位置啊。客户：宝庆东路这里邵阳市宝庆东路这边？坐席：宝庆东路是吗客户：那那个嗯？坐席：噢庆东路这个是横向的吗？一条路，大祥区宝庆中路，有还有更具体点的，有宝庆中路多少号或者是什么标志的建筑，有没有。客户：这个月就是好那。嗯好？坐席：宝兴东路是吗？客户：一会把新郑路口4号大路交界口，这里这里有个什么是？坐席：大庆东路听到了，后面那个路是什么路啊。客户：嗯2号二手车就是市场？坐席：什么市场。客户：Q呃。坐席：什么小市市场先生客户：啊啊？坐席：大汉大小的大吗。客户：多大？啊它就是出汗的话，就3点水加个月8号，二手车交易市场这一块。好？坐席：那您的话。打接都有问题是吗。客户：然后也没有，然后我到邵东，邵东是我那个是乡，镇也有问题也打不通就是？坐席：过了吗？重启都试过了吗，客户：重启好，重试过，坐席：留一个其他的联系方式啊。客户：知道用流量给我，就是打不通电话，别人也打不通，我也打。坐席：流量确实很差，断断续续的我声音都听不太清，留一个联系电话，不要留本机，有别的号码，客户：嗯接电话。坐席：联系电话留多客户：155？坐席：要多少客户：我是这个号码，嗯我是。坐席：您的联系电话，这个电话真听不清，请您报清楚一下联系方式，对。客户：03551多少455？坐席：噢您以后如果出现这种本机号码的声音信号问题的话，您最好不要用故障号码打，因为真的沟通太难了，我听不太清妖55是吧？客户：是15815，我有其他的吗？你们联通？坐席：是的，我是说您如果以后有这种手，就是这个号码，有问题尽可能用别的号码打然后报这个号码的故障就行了，因为这样子方便沟通，您用那种听不清的号码打我，们这边交流很困难，155后面是多少。客户：4770.多少7670？坐席：74是吗客户：76？坐席：.76.76什么。客户：今天已经唉呀。坐席：后面呢客户：8400多少？2400？坐席：.啊嗯我再重新念一下，我不知道对不对，啊因为声音真的非常不清楚，155那你说7670，然后是8400是不是这个客户：是是吗？呃？坐席：不对是吗。客户：对对对，我年初的人打我人人？坐席：是对，还是不对先生我听不清听是对的吗。客户：你好？坐席：火车站那边啊1557670，然后是8400这个做个联系电话是这个链接吗？是不？是客户：经经经营商。噢？坐席：他沟通太困难了。我我再我再我再跟您确认一下，您刚刚报的联系电话是不是1557670，然后是84001的对吗。客户：对对对。坐席：确定是这个，请您保持电话。客户：对对对？坐席：对，技术专家给您回电，您注意接听好吧客户：好好好谢谢好，祝您。"
# text = "坐席：您好，很高兴为您服务您好，客户：唉你好，我刚才办了一个咱们这边的卡，我交了一下费，你看我怎么从这个手机APP上看不到。坐席：那您那个手机号码是多少，呢我帮您再查询一下。客户：1908354838，我给你发了，联盟发了坐席：908354客户：.啊？坐席：好，11083548385是吧。客户：对对对，就放了？坐席：您的这个户主名字叫什么。客户：梁红兵？坐席：身份证的后6位是多少。客户：您说什么什么122875？你给我说明？坐席：322875是吧客户：122875.我。坐席：好。客户：我再看哈。没有了坐席：这边帮您这边查，看了一下，看一下您刚发的是有充值的，这个50块钱，预览显示是到账了，先生看来您是要是？客户：不到账一。不是就是这个是50送120那个对不对。坐席：对是的，显示已经到账了，先生，那个50块钱。客户：嗯行行行。坐席：然后是在10点过8分到账的，您说？客户：啊每月返10块对吧。坐席：对，是的，从现在开始，每个月给您返了10块钱，给您返还12个月。客户：怎么了？行行行？坐席：然后呢你下月的话月租就是19块钱月，那您这边还有什么呃其他问题吗。客户：没有了，谢谢你好的话坐席：那麻烦您稍后个人服务作出评价，谢谢您。"
text = text.replace("\n", "").replace("\r", "").replace("\t", "")

labels = []

bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)

text = text_between_customer(text)
# 将句子转换成向量
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

# 模型预测
predicted = load_model.predict(x_train)[0]

indices = sorted(
    [i for i in range(len(predicted)) if predicted[i] > 0.2],
    key=lambda i: predicted[i],
    reverse=True,
)[:5]
# 可以设置五级，输出其标签  0.8-0.7-0.6-0.5-0.5

with open("model/rinse/cnn/event_type.json", "r", encoding="utf-8") as g:
    movie_genres = json.loads(g.read())

print("预测语句: %s" % text)
print("预测事件类型: %s" % "-".join([movie_genres[index] for index in indices]))
