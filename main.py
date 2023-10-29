"""没用的以前的文件"""
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# from neural_netwaork import DealDataset, Fullconnection

df = pd.read_csv('../Data_s/20000_std_5.csv')
data = df.to_numpy()  # 删除原始文件的第一行列名

# 参数
epoch = 10

# random_state:随机初始化状态，可以保证每次运行程序选择出的集合是一样的
train_data, test_data = train_test_split(data, test_size=0.3, random_state=4, shuffle=True)

train_set = DealDataset(train_data)
test_set = DealDataset(test_data)
# 每次迭代batch_size个数据
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# 训练网络
net = Fullconnection(13, 5)
net = net.double()  # 必须加这一句，不然报错
net.start_train(train_loader, test_data=test_loader, epochs=10, lr=0.001,
                loss_figure=False)

net.evaluate(test_loader)
net.save_module('module.pt')
