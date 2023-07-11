# Bert-Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本多标签分类，Bert，ERNIE，基于pytorch，开箱即用。

## 介绍
模型介绍、数据流动过程：<https://comdy.blog.csdn.net/article/details/125949887#t1>

机器：一块3060

## 环境
python 3.11  
pytorch 2.0。1  
tqdm  
sklearn  
tensorboardX  



### 更换自己的数据集
 - 按照我数据集的格式来格式化你的中文数据集。  


## 效果

模型|acc|备注
--|--|--
bert|94.83%|单纯的bert
ERNIE|94.61%|说好的中文碾压bert呢  
bert_CNN|94.44%|bert + CNN  
bert_RNN|94.57%|bert + RNN  
bert_RCNN|94.51%|bert + RCNN  
bert_DPCNN|94.47%|bert + DPCNN  

原始的bert效果就很好了，把bert当作embedding层送入其它模型，效果反而降了，之后会尝试长文本的效果对比。

CNN、RNN、DPCNN、RCNN、RNN+Attention、FastText等模型效果，请见我另外一个[仓库](https://github.com/649453932/Chinese-Text-Classification-Pytorch)。  

## 预训练语言模型
bert模型放在 bert-base-chinese目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   


ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip  
来自[这里](https://github.com/nghuyong/ERNIE-Pytorch)  

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
![Alt text](image/md/image.png)

1. bert-base-chinese：bert的预训练文件；
2. model：存放bert模型代码；
3. datas：data存放数据集；log 保存 run 时运行日志
4. run.py：项目运行主程序；
5. utils.py：处理数据集并且预加载；
6. train_eval.py：模型训练、验证、测试代码
7. cut.py：划分测试集、验证集、训练集脚本

[1] 数据集文件的构成
![!\[Alt text\](image.png)](image/md/image.png)
其中，class1.csv是原数据集文件，训练集train.csv、验证集dev.csv、测试集test.csv是之前拆分好的，class.txt是标签目录，label.pkl是压缩存储的标签，方便快速读取用的。

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  



## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
