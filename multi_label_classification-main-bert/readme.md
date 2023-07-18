# 基于bert多标签分类

修改使其能够在Python3.11上运行

# 步骤
- 数据样例见data/train.json
- 数据清洗，数据集划分 data_cleanr.py
- 获得label2idx.json python data_preprocess.py
- 训练 python train.py
- 预测 pyhton predict.py
