

import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomNet


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device):
    """定义训练流程。
    :param epoch: 定义训练的总轮次
    :param dataloader: 数据加载器
    :param model: 模型，需在model.py文件中定义好
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备，即使用哪一块CPU、GPU进行训练
    """
    # 将模型置为训练模式
    model.train()

    # START----------------------------------------------------------
    for i in range(epoch):
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            # 将数据和目标移动到正确的设备
            data, targets = data.to(device), targets.to(device)

            # 前向传播
            outputs = model(data)
            loss = loss_fn(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除之前累积的梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()
            # 打印训练信息（可选，每多少个batch打印一次）
        ave_loss = total_loss / len(dataloader)
        print(f"Epoch {i + 1}/{epoch}, Loss: {ave_loss:.4f}")

                # END----------------------------------------------------

    # 注意：通常不建议在每次epoch结束时都保存模型，因为这可能会占用大量磁盘空间。
    # 这里为了演示，我保留了原始代码中的保存操作，但在实际应用中应该根据验证集的性能来决定是否保存模型。
    # 保存模型（通常是根据验证集的性能来决定是否保存）
    torch.save(model, './models/model.pkl')

    # 如果想要保存模型，可以考虑保存模型的状态字典（state_dict）
    # torch.save(model.state_dict(), './models/model_epoch_{}.pth'.format(epoch))


if __name__ == "__main__":
    # 定义模型超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCH = 100

    # 模型实例化
    model = CustomNet()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 训练数据加载器
    train_dataloader = DataLoader(CustomDataset('C:\\nndl_project\\nndl_project-master\\images\\train.txt', 'C:\\nndl_project\\nndl_project-master\\images\\train', ToTensor),
                                  batch_size=BATCH_SIZE)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 学习率和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # 调用训练方法
    train_loop(EPOCH, train_dataloader, model, loss_fn, optimizer, device)
