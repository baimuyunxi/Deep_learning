import torch
from torch import nn

print(torch.cuda.is_available()) # true 查看GPU是否可用

print(torch.cuda.device_count()) #GPU数量， 1

print(torch.cuda.current_device()) #当前GPU的索引， 0

print(torch.cuda.get_device_name(0)) #输出GPU名称