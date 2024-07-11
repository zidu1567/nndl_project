# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               ★★★请在空白处填写适当的语句，将CustomNet类的定义补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F  # Importing torch.nn.functional as F

import torch
from torch import nn


class CustomNet(nn.Module):
    """自定义神经网络模型。
    请完成对__init__、forward方法的实现，以完成CustomNet类的定义。
    """

    def __init__(self):
        """初始化方法。
        在本方法中，请完成神经网络的各个模块/层的定义。
        请确保每层的输出维度与下一层的输入维度匹配。
        """
        super(CustomNet, self).__init__()

        # START----------------------------------------------------------
        # 第一个卷积层，输入通道数为3（彩色图片），输出通道数为16，卷积核大小为3x3，步长为1，填充为1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 第一个池化层，池化核大小为2x2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为3x3，步长为1，填充为1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 第二个池化层，池化核大小为2x2，步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 第三个池化层，池化核大小为2x2，步长为2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 假设我们有10个分类，所以输出层有10个神经元
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 注意这里要计算经过三次池化后的特征图大小
        self.fc2 = nn.Linear(512, 10)
        # END----------------------------------------------------

    def forward(self, x):
        """前向传播过程。
        在本方法中，请完成对神经网络前向传播计算的定义。
        """
        # START----------------------------------------------------------
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # 展平操作，将多维的输入一维化，以便输入全连接层
        x = x.view(-1, 64 * 8 * 8)  # 计算卷积和池化后的特征图大小

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # END----------------------------------------------------
        return x

    # 注意：由于我们还没有导入F（即torch.nn.functional），所以需要在forward方法顶部添加


# import torch.nn.functional as F
# 但为了保持代码整洁，建议将import语句放在文件顶部

# 现在你的CustomNet类已经准备好了，可以继续进行测试
# ...（测试代码与之前相同）


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('C:\\nndl_project\\nndl_project-master\\images\\train.txt', 'C:\\nndl_project\\nndl_project-master\\images\\\\train', ToTensor)
    net = CustomNet()                                # 实例化
    x = torch.unsqueeze(c[10]['image'], 0)      # 模拟一个模型的输入数据
    print(net.forward(x))                            # 测试forward方法

