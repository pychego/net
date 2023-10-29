import os
import torch
import argparse
import pandas as pd
import numpy as np
import os.path as osp
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from utils import AverageMeter


# local imports
from model import FC
from dataset import DealDataset, TimeSeqDataset
from logger import create_logger
current_path = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
# model setting
# 数据标准化对训练正确性影响极大!!!!!!!!!
parser.add_argument("--input_dim", type=int, default=784) # 16 features
parser.add_argument("--classes", type=int, default=10) # 6 classes


# train_data = np.loadtxt('../data/MNIST/train.csv', delimiter=',')
# test_data = np.loadtxt('../data/MNIST/test.csv', delimiter=',')

path = '../data/MNIST/mnist.npz' #数据路径
f = np.load(path) #加载数据
x_train, y_train = f['x_train'], f['y_train'] #导入训练集的输入和标签
print(x_train.shape)
print(y_train.shape)
x_test, y_test = f['x_test'], f['y_test']  #导入测试集的输入和标签
f.close()
# 将(60000, 28, 28)变成(60000, 784)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# 合并x_train和y_train
train_data = np.hstack((x_train, y_train.reshape(-1, 1)))
test_data = np.hstack((x_test, y_test.reshape(-1, 1)))





# 选择小样本训练和测试
num_train = train_data.shape[0]
selected_rows = np.random.choice(num_train, size=int(1.0 * num_train), replace=False)
train_data = train_data[selected_rows, :]
# 取出train_data的前784列进行标准化
x_train = train_data[:, :784]
scaler = StandardScaler(copy=False)
scaler.fit(x_train)
scaler.transform(x_train)  # 标准归一化
train_data = np.hstack((x_train, train_data[:, 784].reshape(-1, 1)))

num_test  = test_data.shape[0]
selected_rows = np.random.choice(num_test, size=int(1.0 * num_test), replace=False)
test_data = test_data[selected_rows, :]
x_test = test_data[:, :784]
scaler = StandardScaler(copy=False)
scaler.fit(x_test)
scaler.transform(x_test)  # 标准归一化
test_data = np.hstack((x_test, test_data[:, 784].reshape(-1, 1)))


parser.add_argument("--train_dataset", default=train_data)
parser.add_argument("--test_dataset", default=test_data)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=0)

# optimizer setting
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=200)

# global setting
parser.add_argument("--ckpt", type=str, default="./ckpt")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--save_per_epoch", type=int, default=45)

opt, unknown = parser.parse_known_args()

device = torch.device(opt.device)
losses = []
os.makedirs(opt.ckpt, exist_ok=True) # 创建文件夹


def main():
    print(current_path)
    print("Arguments: \n{}".format(
        "\n".join(["{}: {}".format(k, v) for k, v in dict(vars(opt)).items()])
    ))
    # setup model
    model = FC(opt.input_dim, opt.classes).to(device)

    # train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    # train_data = train_data.values
    # test_data = test_data.values
    train_dataset = DealDataset(train_data)
    test_dataset = DealDataset(test_data)
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0,
    )

    # setup optimizer
    # optimizer = SGD(autoencoder.parameters(), lr=opt.lr, momentum=.999)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    epoch = 0
    
    # setup loss
    loss_func = nn.CrossEntropyLoss()

    # starting training
    max_acc = 0
    max_epoch = 0
    for epoch in tqdm(range(epoch, opt.epochs)):
        train(model, train_loader, optimizer, loss_func)
        acc = test(model, test_loader)
        if acc > max_acc:
            ckpt = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt["model"], osp.join(opt.ckpt, "fc-{:04d}.ckpt".format(epoch)))
            max_acc = acc
            max_epoch = epoch
        torch.save(ckpt, osp.join(opt.ckpt, "latest.ckpt"))
    print(f"Max Acc: {max_acc} at {max_epoch}")
    # 保存损失
    torch.save(losses, "loss.pt")
    # max_epoch = 167
    ckpt = torch.load(os.path.join(opt.ckpt, "fc-{:04d}.ckpt".format(max_epoch)), map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    evaluate(model, test_loader)

def train(model, train_loader, optimizer, loss_func):
    model.train()
    # print(f"======== Training epoch {epoch} ========")
    # 损失列表
    loss_meter = AverageMeter()
    input_map = lambda x: x.float().to(device)
    for data, label in train_loader:
        data, label = map(input_map, (data, label))

        output = model(data)

        loss = loss_func(output, label.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    losses.append(loss_meter.avg)
        

@torch.no_grad()
def test(model, test_loader):
    """评估模型
    - 参数:\n
        test_data - 测试集\n
        return: 返回模型在测试集的分类正确率
    """
    accuracy, total = 0, 0
    y_true = np.array([])
    y_pred = None
    input_map = lambda x: x.float().to(device)
    for row, label in test_loader:
        row, label = map(input_map, (row, label))
        # # label = label.reshape(-1).long()
        # output = model(row)
        # y_true = np.concatenate([y_true, label.cpu().numpy()])
        # if y_pred is None: y_pred = output.cpu().numpy()
        # else:
        #     y_pred = np.concatenate([y_pred, output.cpu().numpy()])


        # total += len(row)
        # # total += label.shape[0]
        # accuracy += ((output.argmax(dim=1) - label.squeeze()) == 0).sum()
        row, label = map(input_map, (row, label))
        # label = label.reshape(-1).long()
        # 得到神经网络的原始输出
        output = model(row)
        y_true = np.concatenate([y_true, label.cpu().numpy()])
        if y_pred is None: y_pred = output.argmax(dim=1).cpu().numpy()
        else:
            y_pred = np.concatenate([y_pred, output.argmax(dim=1).cpu().numpy()])


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # print(type(accuracy.numpy()))
    acry = np.sum(np.diag(cm)) / np.sum(cm[1:,:])
    # print(f'test sum: {str(total)}')
    print(f'\n test accuracy: {str(acry)}')
    return acry


@torch.no_grad()
def evaluate(model, test_loader):
    # 计算混淆矩阵
    y_true = np.array([])
    y_pred = None
    input_map = lambda x: x.float().to(device)
    for row, label in test_loader:
        row, label = map(input_map, (row, label))
        # label = label.reshape(-1).long()
        # 得到神经网络的原始输出
        output = model(row)
        y_true = np.concatenate([y_true, label.cpu().numpy()])
        if y_pred is None: y_pred = output.argmax(dim=1).cpu().numpy()
        else:
            y_pred = np.concatenate([y_pred, output.argmax(dim=1).cpu().numpy()])


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('混淆矩阵:\n')
    print(cm)
    # 计算准确率
    # 把cm的后四行所有元素加起来

    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], digits=3)
    print(report)

    all_at = np.sum(cm[1:,:])
    right_at = np.sum(np.diag(cm))   # 对角线元素之和
    at_accuracy = right_at / all_at
    print('最终准确率{}'.format(at_accuracy))  

if __name__ == '__main__':
    main()