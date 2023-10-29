from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
import os.path as osp

from tqdm import tqdm
from torch import nn
from torch.optim import Adam, SGD
import torch
from torch.utils.data import DataLoader
from dataset import DealDataset
from dataset import DealDataset
from model import FC
from train import opt, train, test, evaluate


# 读取数据
data = pd.read_csv('../Dataset10/at_add.csv')
# convert data to numpy array
data = data.values  
# select columns 1 to 13 from data  
X = data[:, 1:16]
y = data[:, 16]


# ====神经网络====
# read 20000.csv using pandas
net = FC(opt.input_dim, opt.classes)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# train_data = train_data.values
# test_data = test_data.values
train_dataset = DealDataset(train_data)
test_dataset = DealDataset(test_data)
train_loader = DataLoader(
    train_dataset, batch_size=opt.batch_size, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=opt.batch_size, shuffle=True
)
# setup optimizer
# optimizer = SGD(autoencoder.parameters(), lr=opt.lr, momentum=.999)
optimizer = Adam(net.parameters(), lr=opt.lr)

epoch = 0
losses = []

# setup loss
loss_func = nn.CrossEntropyLoss()

# starting training
max_acc = 0
max_epoch = 0
for epoch in tqdm(range(epoch, opt.epochs)):
    train(net, train_loader, optimizer, loss_func)
    acc = test(net, test_loader)
    if acc > max_acc:
        ckpt = {
            "model": net.state_dict(),
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
net.load_state_dict(ckpt)
net.eval()
evaluate(net, test_loader)



# ====knn====
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)






# 第一层分类器
models = [('svm', SVC(probability=True)), ('rf', RandomForestClassifier()), ('mlp', MLPClassifier())]

# 用于保存第一层分类器的预测结果
first_layer_train = np.zeros((X_train.shape[0], len(models)))
first_layer_test = np.zeros((X_test.shape[0], len(models)))

# 交叉验证，生成第一层分类器的训练数据
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    for j, (name, model) in enumerate(models):
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        first_layer_train[val_index, j] = y_pred
        y_pred_test = model.predict_proba(X_test)[:, 1]
        first_layer_test[:, j] += y_pred_test / kf.n_splits

# 第二层分类器
second_layer_model = LogisticRegression()
second_layer_model.fit(first_layer_train, y_train)
y_pred = second_layer_model.predict(first_layer_test)

# 输出结果
print('Accuracy:', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5])
# print('混淆矩阵:\n')
print(cm)
# 计算准确率
accuracy = np.trace(cm) / np.sum(cm)
report = classification_report(y_test, y_pred, digits=4, labels=[0, 1, 2, 3, 4, 5])
print(report)
