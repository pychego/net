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
from sklearn.metrics import confusion_matrix, classification_report

from utils import AverageMeter


# local imports
from model import TimeSequence, FC
from dataset import DealDataset, TimeSeqDataset
from logger import create_logger
current_path = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
# model setting
# 数据标准化对训练正确性影响极大!!!!!!!!!
parser.add_argument("--input_dim", type=int, default=16) # 16 features
parser.add_argument("--classes", type=int, default=6) # 6 classes

# dataset setting 修改数据集和Dataset类, 输入类别
data = pd.read_csv('Dataset10/at_add.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
parser.add_argument("--train_dataset", default=train_data)
parser.add_argument("--test_dataset", default=test_data)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=0)

# optimizer setting
parser.add_argument("--lr", type=float, default=0.005)
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

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    train_data = train_data.values
    test_data = test_data.values
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
        # label = label.reshape(-1).long()
        output = model(row)
        y_true = np.concatenate([y_true, label.cpu().numpy()])
        if y_pred is None: y_pred = output.cpu().numpy()
        else:
            y_pred = np.concatenate([y_pred, output.cpu().numpy()])

        total += len(row)
        # total += label.shape[0]
        accuracy += ((output.argmax(dim=1) - label.squeeze()) == 0).sum()
    
    # print(type(accuracy.numpy()))
    acry = accuracy / total
    # print(f'test sum: {str(total)}')
    print(acry.item())
    return acry


@torch.no_grad()
def evaluate(model, test_loader):
    # 计算混淆矩阵
    y_true = np.array([])
    y_pred = None
    y_output = None
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


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    print('混淆矩阵:\n')
    print(cm)
    # 计算准确率
    # 把cm的后四行所有元素加起来

    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], digits=3)
    print(report)

    all_at = np.sum(cm[1:,:])
    right_at = cm[1][1] + cm[2][2] + cm[3][3] + cm[4][4] + cm[5][5]
    at_accuracy = right_at / all_at
    print(at_accuracy)  

if __name__ == '__main__':
    main()