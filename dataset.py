import torch
import numpy
from torch.utils.data import Dataset

"""将数据包装成Dataset类

Returns:
    _type_: _description_
"""

class DealDataset(Dataset):
    def __init__(self, data: numpy.ndarray):
        super().__init__()
        # 发送时间, 传输时间, 发送者位置和速度, 接受者位置和速度, 消息ID
        # remove the 13th column
        # convert to tensor
        self.x = torch.from_numpy(data[:, 0:784])  # 前13列是特征
        # self.x = numpy.delete(data, 13, axis=1)
        self.label = torch.from_numpy(data[:, 784])  # 第14列是标签


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index] , self.label[index]

class TimeSeqDataset(Dataset):

    def __init__(self, data: numpy.ndarray, seqlen: int = 25, stride: int = 10):
        super().__init__()
        # 发送时间, 传输时间, 发送者位置和速度, 接受者位置和速度, 消息ID
        # remove the 13th column
        self.seqlen = seqlen
        self.stride = stride
        self.x = numpy.delete(data, 13, axis=1)
        self.label = torch.from_numpy(data[:, 13])  # 第14列是标签


    def __len__(self):
        return (len(self.x) - self.seqlen) // self.stride

    def __getitem__(self, index):
        # read the data sequence with length seqlen, and stride stride
        # return the data sequence and the label of the last data
        if index + self.seqlen > len(self.x):
            return None
        return self.x[index * self.stride: index * self.stride + self.seqlen], \
            self.label[index * self.stride: index * self.stride + self.seqlen] // 2