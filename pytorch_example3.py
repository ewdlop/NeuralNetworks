"""
PyTorch 快速入门教程

本教程是PyTorch的完整入门指南，涵盖了机器学习项目的主要步骤：
1. 数据加载和预处理
2. 模型定义和架构设计
3. 训练循环和优化
4. 模型评估和测试
5. 模型保存和加载
6. 实际预测应用

学习目标：
- 掌握PyTorch的基本工作流程
- 理解深度学习项目的完整生命周期
- 学会数据处理、模型训练、评估的标准做法
- 能够独立完成一个完整的分类项目

`Learn the Basics <intro.html>`_ ||
**Quickstart** ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

快速入门
===================
本节介绍机器学习常见任务的API。请参考每节中的链接以深入了解。

处理数据
-----------------
PyTorch有两个 `处理数据的基本组件 <https://pytorch.org/docs/stable/data.html>`_：
``torch.utils.data.DataLoader`` 和 ``torch.utils.data.Dataset``。
``Dataset`` 存储样本及其对应的标签，``DataLoader`` 在 ``Dataset`` 周围包装一个可迭代对象。

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

######################################################################
# 数据集获取和加载
# PyTorch提供特定领域的库，如 `TorchText <https://pytorch.org/text/stable/index.html>`_、
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_ 和 `TorchAudio <https://pytorch.org/audio/stable/index.html>`_，
# 所有这些都包含数据集。在本教程中，我们将使用TorchVision数据集。
#
# ``torchvision.datasets`` 模块包含许多真实世界视觉数据的 ``Dataset`` 对象，如
# CIFAR、COCO（`完整列表在这里 <https://pytorch.org/vision/stable/datasets.html>`_）。
# 在本教程中，我们使用FashionMNIST数据集。每个TorchVision ``Dataset`` 都包含两个参数：
# ``transform`` 和 ``target_transform`` 分别用于修改样本和标签。

print("="*60)
print("步骤1：加载和准备数据")
print("="*60)

# 从开放数据集下载训练数据
# FashionMNIST：时尚物品的图像分类数据集，包含10个类别
print("正在下载训练数据...")
training_data = datasets.FashionMNIST(
    root="data",           # 数据存储路径
    train=True,            # 下载训练集
    download=True,         # 如果数据不存在则下载
    transform=ToTensor(),  # 将PIL图像或numpy数组转换为张量
)

# 从开放数据集下载测试数据
print("正在下载测试数据...")
test_data = datasets.FashionMNIST(
    root="data",           # 数据存储路径
    train=False,           # 下载测试集
    download=True,         # 如果数据不存在则下载
    transform=ToTensor(),  # 数据类型转换
)

print(f"训练样本数量: {len(training_data)}")
print(f"测试样本数量: {len(test_data)}")
print(f"图像尺寸: {training_data[0][0].shape}")  # (C, H, W)
print(f"类别数量: {len(training_data.classes)}")
print(f"类别名称: {training_data.classes}")

######################################################################
# 创建数据加载器
# 我们将 ``Dataset`` 作为参数传递给 ``DataLoader``。这在我们的数据集上包装了一个可迭代对象，
# 并支持自动批处理、采样、洗牌和多进程数据加载。这里我们定义批次大小为64，
# 即数据加载器可迭代对象中的每个元素将返回一批64个特征和标签。
#
# DataLoader的重要作用：
# 1. 批处理：将单个样本组合成批次，提高训练效率
# 2. 洗牌：随机化样本顺序，避免模型学到数据顺序的偏见
# 3. 并行加载：使用多进程加速数据加载
# 4. 内存管理：按需加载数据，避免内存溢出

batch_size = 64  # 每批处理64个样本

print(f"\n使用批次大小: {batch_size}")
print("批次大小的影响：")
print("- 较大批次：训练稳定，但需要更多内存")
print("- 较小批次：内存友好，但可能训练不稳定")

# 创建数据加载器
train_dataloader = DataLoader(
    training_data, 
    batch_size=batch_size,
    shuffle=True,      # 训练时洗牌数据
    num_workers=2      # 使用2个进程并行加载数据
)
test_dataloader = DataLoader(
    test_data, 
    batch_size=batch_size,
    shuffle=False,     # 测试时不需要洗牌
    num_workers=2
)

# 查看数据的形状和类型
print("\n数据批次信息：")
for X, y in test_dataloader:
    print(f"特征张量 X 的形状 [N, C, H, W]: {X.shape}")
    print(f"- N (批次大小): {X.shape[0]}")
    print(f"- C (通道数): {X.shape[1]} (灰度图像)")
    print(f"- H (高度): {X.shape[2]} 像素")
    print(f"- W (宽度): {X.shape[3]} 像素")
    print(f"标签张量 y 的形状: {y.shape}, 数据类型: {y.dtype}")
    print(f"标签范围: {y.min()} - {y.max()} (对应10个类别)")
    break

######################################################################
# 了解更多关于 `在PyTorch中加载数据 <data_tutorial.html>`_ 的内容。
#

######################################################################
# --------------
#

################################
# 创建模型
# ------------------
print("\n" + "="*60)
print("步骤2：定义神经网络模型")
print("="*60)

# 在PyTorch中定义神经网络，我们创建一个继承自
# `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ 的类。
# 我们在 ``__init__`` 函数中定义网络的层，并在 ``forward`` 函数中指定数据如何通过网络传递。
# 为了加速神经网络中的操作，我们将其移动到 `加速器 <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# 如CUDA、MPS、MTIA或XPU。如果当前加速器可用，我们将使用它。否则，我们使用CPU。

# 选择计算设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"选择的计算设备: {device}")

if device == "cpu":
    print("提示：正在使用CPU训练，速度可能较慢")
    print("如果有GPU，请确保安装了CUDA版本的PyTorch")
else:
    print(f"太好了！正在使用 {device} 加速训练")

# 定义模型类
class NeuralNetwork(nn.Module):
    """
    简单的全连接神经网络用于图像分类
    
    网络架构：
    输入(28x28图像) -> 展平 -> 全连接层(784->512) -> ReLU -> 
    全连接层(512->512) -> ReLU -> 输出层(512->10)
    
    这是一个经典的多层感知机(MLP)，适合入门学习
    """
    def __init__(self):
        """初始化网络层"""
        super().__init__()
        
        # 展平层：将2D图像转换为1D向量
        # (batch_size, 1, 28, 28) -> (batch_size, 784 = 28*28)
        self.flatten = nn.Flatten()
        
        # 定义线性层堆叠（全连接网络）
        self.linear_relu_stack = nn.Sequential(
            # 第一个隐藏层：784个输入 -> 512个神经元
            nn.Linear(28*28, 512),   # 28*28=784 (展平后的图像像素)
            nn.ReLU(),               # ReLU激活函数：f(x) = max(0,x)
            
            # 第二个隐藏层：512 -> 512
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # 输出层：512 -> 10 (10个类别)
            nn.Linear(512, 10)       # 不加激活函数，输出原始logits
        )

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入图像张量 (batch_size, 1, 28, 28)
            
        Returns:
            logits: 分类得分 (batch_size, 10)
        """
        # 步骤1：展平图像
        x = self.flatten(x)
        # 步骤2：通过全连接层
        logits = self.linear_relu_stack(x)
        return logits

# 创建模型实例并移动到选定设备
model = NeuralNetwork().to(device)

print(f"\n模型架构：")
print(model)

# 统计模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型统计：")
print(f"总参数数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

######################################################################
# 了解更多关于 `在PyTorch中构建神经网络 <buildmodel_tutorial.html>`_ 的内容。
#


######################################################################
# --------------
#


#####################################################################
# 优化模型参数
# ----------------------------------------
print("\n" + "="*60)
print("步骤3：定义损失函数和优化器")
print("="*60)

# 要训练模型，我们需要一个 `损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# 和一个 `优化器 <https://pytorch.org/docs/stable/optim.html>`_。
#
# 损失函数：衡量模型预测与真实标签的差距
# 优化器：根据损失函数的梯度更新模型参数

# 选择损失函数
# CrossEntropyLoss：多分类任务的标准选择
# 它结合了LogSoftmax和NLLLoss，数值上更稳定
loss_fn = nn.CrossEntropyLoss()
print("损失函数: CrossEntropyLoss")
print("- 适用于多分类任务")
print("- 自动应用softmax转换")
print("- 数值稳定性好")

# 选择优化器
# SGD：随机梯度下降，经典且可靠的优化算法
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print(f"\n优化器: SGD")
print(f"学习率: {1e-3}")
print("SGD特点：")
print("- 简单可靠")
print("- 收敛稳定")
print("- 适合入门学习")


#######################################################################
# 训练函数
# 在单个训练循环中，模型对训练数据集进行预测（分批输入），
# 并反向传播预测误差来调整模型的参数。
#
# 训练循环的标准步骤：
# 1. 前向传播：计算预测结果
# 2. 计算损失：比较预测和真实标签
# 3. 反向传播：计算梯度
# 4. 参数更新：根据梯度更新权重
# 5. 清零梯度：为下次迭代准备

def train(dataloader, model, loss_fn, optimizer):
    """
    训练函数：执行一个epoch的训练
    
    Args:
        dataloader: 训练数据加载器
        model: 要训练的模型
        loss_fn: 损失函数
        optimizer: 优化器
    """
    # 获取数据集大小用于进度显示
    size = len(dataloader.dataset)
    
    # 设置模型为训练模式
    # 这会启用Dropout、BatchNorm等训练时的特殊行为
    model.train()
    
    # 遍历每个批次
    for batch, (X, y) in enumerate(dataloader):
        # 将数据移动到计算设备
        X, y = X.to(device), y.to(device)

        # 前向传播：计算预测
        pred = model(X)
        
        # 计算损失
        loss = loss_fn(pred, y)

        # 反向传播：计算梯度
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 清零梯度（PyTorch中梯度会累积）
        optimizer.zero_grad()

        # 每100个批次打印一次训练状态
        if batch % 100 == 0:
            loss_value = loss.item()  # 获取损失的数值
            current = (batch + 1) * len(X)  # 当前处理的样本数
            print(f"损失: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# 测试函数
# 我们还要检查模型在测试数据集上的性能，以确保它正在学习。

def test(dataloader, model, loss_fn):
    """
    测试函数：评估模型性能
    
    Args:
        dataloader: 测试数据加载器
        model: 要评估的模型  
        loss_fn: 损失函数
    """
    size = len(dataloader.dataset)  # 测试集大小
    num_batches = len(dataloader)   # 批次数量
    
    # 设置模型为评估模式
    # 这会禁用Dropout、BatchNorm等训练时的随机性
    model.eval()
    
    test_loss, correct = 0, 0
    
    # 禁用梯度计算（节省内存和计算）
    with torch.no_grad():
        for X, y in dataloader:
            # 将数据移动到计算设备
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            pred = model(X)
            
            # 累积损失
            test_loss += loss_fn(pred, y).item()
            
            # 计算准确率
            # pred.argmax(1)：获取每行最大值的索引（预测类别）
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= num_batches
    correct /= size
    print(f"测试结果: \n 准确率: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")

##############################################################################
# 执行训练循环
print("\n" + "="*60)
print("步骤4：开始训练模型")
print("="*60)

# 训练过程在几次迭代（*epochs*）中进行。在每个epoch中，模型学习
# 参数以做出更好的预测。我们在每个epoch打印模型的准确率和损失；
# 我们希望看到准确率随着每个epoch的增加而增加，损失减少。

epochs = 5  # 训练轮数
print(f"开始训练，总共 {epochs} 个epoch")
print("Epoch说明：")
print("- 每个epoch代表模型看过所有训练数据一次")  
print("- 通常需要多个epoch才能收敛")
print("- 可以通过验证集表现来决定何时停止训练")

for t in range(epochs):
    print(f"Epoch {t+1}")
    print("-------------------------------")
    
    # 执行训练
    train(train_dataloader, model, loss_fn, optimizer)
    
    # 执行测试评估
    test(test_dataloader, model, loss_fn)

print("训练完成！")

######################################################################
# 了解更多关于 `训练你的模型 <optimization_tutorial.html>`_ 的内容。
#

######################################################################
# --------------
#

######################################################################
# 保存模型
# -------------
print("\n" + "="*60)
print("步骤5：保存和加载模型")
print("="*60)

# 保存模型的常见方法是序列化内部状态字典（包含模型参数）。
# 
# 模型保存的重要性：
# 1. 避免重复训练
# 2. 部署到生产环境
# 3. 实验结果复现
# 4. 迁移学习的基础

print("保存模型状态字典...")
torch.save(model.state_dict(), "model.pth")
print("已保存PyTorch模型状态到 model.pth")

print("\n模型保存说明：")
print("- state_dict(): 包含所有可学习的参数")
print("- .pth 是PyTorch模型文件的常用扩展名")
print("- 这种方式只保存参数，不保存模型结构")

######################################################################
# 加载模型
# ----------------------------
print("\n正在演示模型加载...")

# 加载模型的过程包括重新创建模型结构并将状态字典加载到其中。
# 注意：加载模型需要先定义相同的网络架构

# 重新创建模型实例
print("1. 重新创建模型架构...")
model = NeuralNetwork().to(device)

# 加载保存的参数
print("2. 加载保存的参数...")
model.load_state_dict(torch.load("model.pth", weights_only=True))

print("模型加载完成！")
print("注意：weights_only=True 确保只加载权重，提高安全性")

#############################################################
# 使用模型进行预测
print("\n" + "="*60)
print("步骤6：使用模型进行预测")
print("="*60)

# 这个模型现在可以用来进行预测。
# 定义FashionMNIST的类别名称
classes = [
    "T-shirt/top",    # T恤/上衣
    "Trouser",        # 裤子  
    "Pullover",       # 套头衫
    "Dress",          # 连衣裙
    "Coat",           # 外套
    "Sandal",         # 凉鞋
    "Shirt",          # 衬衫
    "Sneaker",        # 运动鞋
    "Bag",            # 包
    "Ankle boot",     # 短靴
]

print(f"FashionMNIST数据集包含以下类别：")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

# 设置模型为评估模式
model.eval()

# 从测试集中取一个样本进行预测
x, y = test_data[0][0], test_data[0][1]  # 第一个测试样本
print(f"\n选择测试样本:")
print(f"图像形状: {x.shape}")
print(f"真实标签: {y} ({classes[y]})")

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    # 将输入移动到设备并添加批次维度
    x = x.to(device)
    
    # 获取模型预测z
    # 将输入张量 x 传递给模型 model，得到预测结果 pred
    pred = model(x)
    
    # 获取预测的类别和置信度
    #pred[0] 是预测的类别得分，argmax(0) 是获取得分最高的类别索
    predicted_idx = pred[0].argmax(0).item() # 获取预测的类别索引
    predicted_class = classes[predicted_idx]
    actual_class = classes[y]
    
    # 计算预测概率
    # softmax 函数将预测得分转换为概率分布，使得所有类别的概率之和为1
    # dim=0 表示按列计算概率 ，dim=1 表示按行计算概率 ，dim=None 表示按所有维度计算概率
    # 这里使用 dim=0 表示按列计算概率，即计算每个类别的概率
    # 返回一个包含每个类别概率的张量s
    probabilities = torch.nn.functional.softmax(pred[0], dim=0)
    confidence = probabilities[predicted_idx].item()
    
    print(f"\n预测结果:")
    print(f'预测类别: "{predicted_class}" (置信度: {confidence:.3f})')
    print(f'实际类别: "{actual_class}"')
    print(f'预测{"正确" if predicted_class == actual_class else "错误"}!')
    
    # 显示所有类别的预测概率
    print(f"\n所有类别的预测概率:")
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        marker = "👈" if i == predicted_idx else "  "
        print(f"{i}: {class_name:12} {prob:.3f} {marker}")

######################################################################
# 了解更多关于 `保存和加载你的模型 <saveloadrun_tutorial.html>`_ 的内容。
#

print("\n" + "="*60)
print("教程总结")
print("="*60)
print("恭喜！你已经完成了PyTorch快速入门教程")
print("\n学到的关键概念：")
print("1. 📊 数据加载：Dataset和DataLoader的使用")
print("2. 🧠 模型定义：继承nn.Module创建神经网络")
print("3. 🎯 训练过程：损失函数、优化器、训练循环")
print("4. 📈 模型评估：准确率计算和性能监控")
print("5. 💾 模型持久化：保存和加载模型状态")
print("6. 🔮 实际应用：使用训练好的模型进行预测")

print("\n下一步建议：")
print("- 尝试不同的网络架构（CNN、RNN等）")
print("- 实验不同的优化器（Adam、AdamW等）")
print("- 学习数据增强技术")
print("- 探索迁移学习")
print("- 研究更复杂的数据集")

print("\n快乐的PyTorch学习之旅！🚀")
