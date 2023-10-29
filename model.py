"""有用的文件, train.py中用的模型在这里面"""
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch.nn.functional as F

class LinearBlock(nn.Module):
    """ 线性层 + 批归一化 + 激活函数的全连接层 """
    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, *args, **kwargs)
        # self.batchnorm = nn.BatchNorm1d(out_features)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyrelu(self.linear(x))


class FC(nn.Module):

    def __init__(self, in_features, out_features):
        """
        :param in_features: 输入的特征维数
        :param out_features: 分类的类别数
        :param loss_func:损失函数，默认为交叉熵损失
        """
        super().__init__()
        self.NUM_CLASSES = out_features
        self.train_loss = []
        self.validation_loss = []
        self.block = nn.Sequential(
            LinearBlock(in_features, 32),
            LinearBlock(32, 64),
            LinearBlock(64, 32),
            LinearBlock(32, 20),
            nn.Linear(20, out_features),
        )

    def forward(self, x):
        x = self.block(x)
        # x = F.softmax(x, dim=1)
        return x


    # 在测试集上的损失
    def validation_loss_func(self, test_data):
        self.eval()
        with torch.no_grad():
            loss_epoch = np.array([])
            for data, label in test_data:
                output = self.forward(data)
                loss = self.loss_func(output, label.squeeze().long())
                loss_epoch = np.append(loss_epoch, loss.detach().cpu().numpy())
        return np.mean(loss_epoch)

    def loss_figure(self, epochs: int, loss, validation):
        """绘制训练集和测试集的loss曲线
        必须要对测试集数据进行平滑处理，窗口长度必须为odd， ployorder多项式次数
        """
        epochs = [i + 1 for i in range(epochs)]
        loss_filter = savgol_filter(loss, window_length=25, polyorder=7, mode='nearest')
        validation_filter = savgol_filter(validation, 25, 7, mode='nearest')
        plt.plot(epochs, loss_filter, label='train')
        plt.plot(epochs, validation_filter, label='validation')
        plt.title('model train vs validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss__内部.jpg', dpi=1000)
        plt.show()


class TimeSequence(nn.Module):

    def __init__(self, features: int, hidden_size=16, num_layers=2, classes: int=2):
        super().__init__()
        self.lstm = nn.RNN(
            input_size=features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear_proj = nn.Sequential(
            LinearBlock(hidden_size, classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear_proj(x)
        x = x.reshape(-1, x.shape[-1])
        return x


if __name__ == '__main__':
    # feature dim, hidden size, layers
    lstm = nn.LSTM(14, 20, 2, batch_first=True)
    # batch, seq, feature
    input = torch.randn(5, 40, 14)
    h0 = torch.randn(2, 5, 20)
    c0 = torch.randn(2, 5, 20)
    output, (hn, cn) = lstm(input, (h0, c0))
    print(output.shape) # batch, seq, 20
    # B, C ; B, C, L