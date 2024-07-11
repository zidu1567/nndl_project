# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备
    """
    model.eval()  # 将模型置为评估（测试）模式

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # 增加一个批次维度
    image_tensor = image_tensor.to(device)  # 将图像张量移动到指定设备

    # 使用模型进行推理
    with torch.no_grad():
        output = model(image_tensor)

        # 处理模型的输出
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    # 指定图片路径
    image_path = "C://nndl_project//nndl_project-master//images//test//signs//img_0006.png"

    # 检查CUDA是否可用，并据此设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    # 总是先加载到CPU上，因为.pkl文件可能包含无法直接加载到GPU的Python对象
    # 如果.pkl文件只包含state_dict，则应该使用model.load_state_dict()来加载
    model = torch.load('C:\\nndl_project\\nndl_project-master\\models\\model.pkl', map_location=torch.device('cpu'))
    model.to(device)  # 现在将模型移动到正确的设备

    # 调用inference函数进行推理
    inference(image_path, model, device)