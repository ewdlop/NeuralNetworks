"""
PyTorch 神经网络构建教程

这个教程展示了如何在 PyTorch 中构建神经网络的基础知识。
包含的主要内容：
1. 神经网络的基本概念
2. 如何定义网络架构
3. 各种层的功能和用法
4. 模型参数的管理

学习目标：
- 理解 nn.Module 的基本用法
- 掌握常用层的功能（Linear, ReLU, Flatten等）
- 学会构建完整的神经网络
- 了解模型参数的访问方法

`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
**Build Model** ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

构建神经网络
========================

神经网络由执行数据操作的层/模块组成。
`torch.nn <https://pytorch.org/docs/stable/nn.html>`_ 命名空间提供了构建神经网络所需的所有构建块。
PyTorch中的每个模块都是 `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ 的子类。
神经网络本身也是一个模块，由其他模块（层）组成。这种嵌套结构使得构建和管理复杂架构变得容易。

在以下部分中，我们将构建一个神经网络来对FashionMNIST数据集中的图像进行分类。

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# 获取训练设备
# -----------------------
# 我们希望能够在 `加速器 <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# 上训练我们的模型，如 CUDA, MPS, MTIA, 或 XPU。如果当前加速器可用，我们将使用它。否则，我们使用CPU。
# 
# 设备选择的重要性：
# - GPU训练比CPU快数十倍
# - 现代深度学习几乎都依赖GPU加速
# - PyTorch会自动处理设备间的数据转移

# 检测并选择最佳可用设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"使用设备: {device}")

##############################################
# 定义神经网络类
# -------------------------
# 我们通过继承 ``nn.Module`` 来定义神经网络，并在 ``__init__`` 中初始化神经网络层。
# 每个 ``nn.Module`` 子类都在 ``forward`` 方法中实现对输入数据的操作。
#
# nn.Module 的核心概念：
# 1. 所有层都必须在 __init__ 中定义
# 2. forward 方法定义了数据的前向传播路径
# 3. 反向传播（梯度计算）由 PyTorch 自动处理
# 4. 模型的参数会被自动追踪和管理

class NeuralNetwork(nn.Module):
    """
    简单的全连接神经网络
    
    网络架构：
    - 输入：28x28的图像（784个像素）
    - 隐藏层1：512个神经元 + ReLU激活
    - 隐藏层2：512个神经元 + ReLU激活  
    - 输出层：10个神经元（对应10个类别）
    
    这是一个经典的多层感知机(MLP)架构
    """
    def __init__(self):
        """
        初始化网络层
        
        在这里定义所有需要的层：
        - Flatten: 将2D图像展平为1D向量
        - Linear layers: 全连接层进行特征变换
        - ReLU: 激活函数引入非线性
        """
        super().__init__()  # 调用父类初始化方法
        
        # 展平层：将2D图像(28x28)转换为1D向量(784)
        # 这是处理图像数据的常见预处理步骤
        self.flatten = nn.Flatten()
        
        # 定义线性层堆叠
        # Sequential容器按顺序执行其中的模块
        self.linear_relu_stack = nn.Sequential(
            # 第一层：784 -> 512
            # 28*28=784 是输入特征数（展平后的图像像素）
            nn.Linear(28*28, 512),
            nn.ReLU(),  # ReLU激活：max(0,x)，引入非线性
            
            # 第二层：512 -> 512  
            # 中间隐藏层，保持相同维度
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # 输出层：512 -> 10
            # 10对应FashionMNIST的10个类别
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)
            
        Returns:
            logits: 未经softmax的原始输出，形状为 (batch_size, 10)
            
        数据流程：
        1. 输入图像 -> 展平为向量
        2. 通过线性层和激活函数
        3. 输出原始分数（logits）
        """
        # 步骤1：展平图像
        # (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = self.flatten(x)
        
        # 步骤2：通过线性层堆叠
        # (batch_size, 784) -> (batch_size, 10)
        logits = self.linear_relu_stack(x)
        
        return logits

##############################################
# 创建模型实例并移动到指定设备
# 
# 重要概念：
# - .to(device) 将模型参数移动到指定设备（GPU/CPU）
# - 模型和数据必须在同一设备上才能进行计算
# - print(model) 显示模型的完整架构

# 创建神经网络实例
model = NeuralNetwork().to(device)
print("模型架构：")
print(model)


##############################################
# 使用模型进行推理
# 
# 要使用模型，我们直接调用模型实例，这会执行模型的 ``forward`` 方法，
# 以及一些 `后台操作 <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_。
# 不要直接调用 ``model.forward()``！
#
# 模型输出解释：
# - 输出是2维张量，dim=0对应批次中的每个样本
# - dim=1对应每个类别的原始预测值（logits）
# - 通过 Softmax 可以将logits转换为概率分布

# 创建一个随机输入张量（模拟一张28x28的图像）
X = torch.rand(1, 28, 28, device=device)
print(f"输入张量形状: {X.shape}")

# 获取模型的原始输出（logits）
logits = model(X)
print(f"模型输出(logits)形状: {logits.shape}")
print(f"原始输出值: {logits}")

# 将logits转换为概率分布
# Softmax函数：将任意实数向量转换为概率分布
pred_probab = nn.Softmax(dim=1)(logits)
print(f"预测概率: {pred_probab}")

# 获取预测类别（概率最大的类别）
y_pred = pred_probab.argmax(1)
print(f"预测类别: {y_pred}")


######################################################################
# 详细解析模型层
# --------------


##############################################
# 模型层详解
# -------------------------
#
# 让我们详细分解FashionMNIST模型中的各个层。为了演示，我们将使用
# 一个包含3张28x28大小图像的小批次，看看数据在网络中的传播过程。

# 创建一个小批次的输入数据（3张图像）
input_image = torch.rand(3, 28, 28)
print(f"输入图像张量形状: {input_image.size()}")

##################################################
# nn.Flatten 展平层
# ^^^^^^^^^^^^^^^^^^^^^^
# 我们初始化 `nn.Flatten <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_
# 层来将每个2D的28x28图像转换为包含784个像素值的连续数组
# （小批次维度在dim=0处保持不变）。
#
# Flatten的作用：
# - 将多维张量展平为一维（保留批次维度）
# - 这是连接卷积层和全连接层的关键步骤
# - 例如：(3, 28, 28) -> (3, 784)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"展平后的图像形状: {flat_image.size()}")
print(f"展平过程：{input_image.shape} -> {flat_image.shape}")

##############################################
# nn.Linear 线性层（全连接层）
# ^^^^^^^^^^^^^^^^^^^^^^
# `线性层 <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
# 是一个模块，使用其存储的权重和偏置对输入应用线性变换。
#
# 线性变换公式：y = xW^T + b
# - W: 权重矩阵 (out_features, in_features)
# - b: 偏置向量 (out_features,)  
# - x: 输入张量 (batch_size, in_features)
# - y: 输出张量 (batch_size, out_features)

# 创建一个线性层：784个输入特征 -> 20个输出特征
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f"线性层输出形状: {hidden1.size()}")
print(f"权重矩阵形状: {layer1.weight.shape}")
print(f"偏置向量形状: {layer1.bias.shape}")


#################################################
# nn.ReLU 激活函数
# ^^^^^^^^^^^^^^^^^^^^^^
# 非线性激活函数是在模型输入和输出之间创建复杂映射的关键。
# 它们在线性变换后被应用，引入 *非线性*，帮助神经网络学习各种复杂现象。
#
# ReLU函数：f(x) = max(0, x)
# - 优点：计算简单，缓解梯度消失问题
# - 作用：引入非线性，使网络能学习复杂模式
# - 替代品：Sigmoid, Tanh, LeakyReLU, GELU等
#
# 在这个模型中，我们在线性层之间使用 `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_

print(f"ReLU激活前的值: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"ReLU激活后的值: {hidden1}")
print("注意：所有负值都变成了0，正值保持不变")



#################################################
# nn.Sequential 序列容器
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ 是一个有序的
# 模块容器。数据按照定义的相同顺序通过所有模块。你可以使用序列容器
# 快速构建网络，如下面的 ``seq_modules``。
#
# Sequential的优势：
# - 代码简洁，易于阅读
# - 自动管理数据流
# - 适合简单的前馈网络
# - 可以嵌套使用

# 使用Sequential构建一个完整的小网络
seq_modules = nn.Sequential(
    flatten,           # 展平层
    layer1,           # 第一个线性层 (784->20)
    nn.ReLU(),        # ReLU激活
    nn.Linear(20, 10) # 第二个线性层 (20->10)
)

# 测试Sequential网络
input_image = torch.rand(3, 28, 28) # 创建一个3张28x28的图像
logits = seq_modules(input_image)
print(f"Sequential网络输出形状: {logits.shape}")
print(f"数据流：{input_image.shape} -> 展平 -> 线性层 -> ReLU -> 线性层 -> {logits.shape}")

################################################################
# nn.Softmax Softmax层
# ^^^^^^^^^^^^^^^^^^^^^^
# 神经网络的最后一个线性层返回 `logits` - 在 [-∞, ∞] 范围内的原始值 - 这些值被传递给
# `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ 模块。
# logits被缩放到 [0, 1] 范围内的值，表示模型对每个类别的预测概率。
# ``dim`` 参数指示沿哪个维度的值必须总和为1。
#
# Softmax函数：σ(z_i) = e^{z_i} / Σ(e^{z_j})
# - 将任意实数向量转换为概率分布
# - 所有输出值在[0,1]之间且总和为1
# - 放大最大值，压缩其他值的影响

# 创建Softmax层
softmax = nn.Softmax(dim=1)  # dim=1表示沿类别维度计算
pred_probab = softmax(logits)

print(f"Softmax前(logits): {logits[0]}")
print(f"Softmax后(概率): {pred_probab[0]}")
print(f"概率总和: {pred_probab[0].sum()}")  # 应该等于1.0

print("\n" + "="*60)
print("深入理解 Softmax 的 dim 参数")
print("="*60)

# 详细解释 dim 参数的含义
print("\n1. dim 参数的基本概念：")
print("   dim 指定了 Softmax 函数应用的维度")
print("   在该维度上，所有值的总和会变成 1.0")

# 创建一个示例张量来演示
example_logits = torch.tensor([
    [2.0, 1.0, 0.1],  # 第一个样本的logits
    [0.5, 3.0, 1.5],  # 第二个样本的logits
    [1.0, 1.0, 4.0]   # 第三个样本的logits
])
print(f"\n2. 示例张量形状: {example_logits.shape}")
print(f"   维度含义: (batch_size=3, num_classes=3)")
print(f"   原始logits:\n{example_logits}")

# dim=1 的情况（沿类别维度）
softmax_dim1 = nn.Softmax(dim=1)
prob_dim1 = softmax_dim1(example_logits)
print(f"\n3. 使用 dim=1 (沿类别维度):")
print(f"   每行(每个样本)的概率总和为1")
print(f"   转换后的概率:\n{prob_dim1}")
print(f"   每行总和: {prob_dim1.sum(dim=1)}")  # 应该都是1.0

# dim=0 的情况（沿批次维度）
softmax_dim0 = nn.Softmax(dim=0)
prob_dim0 = softmax_dim0(example_logits)
print(f"\n4. 使用 dim=0 (沿批次维度):")
print(f"   每列(每个类别)的概率总和为1")
print(f"   转换后的概率:\n{prob_dim0}")
print(f"   每列总和: {prob_dim0.sum(dim=0)}")  # 应该都是1.0

print(f"\n5. 为什么分类任务中使用 dim=1？")
print(f"   - 我们希望每个样本在所有类别上的概率总和为1")
print(f"   - 这样可以解释为该样本属于各个类别的概率")
print(f"   - dim=1 正好满足这个要求")

# 实际应用示例
print(f"\n6. 实际分类示例：")
class_names = ['T-shirt', 'Trouser', 'Pullover']
for i, (logits_sample, probs_sample) in enumerate(zip(example_logits, prob_dim1)):
    predicted_class = probs_sample.argmax().item()
    confidence = probs_sample.max().item()
    print(f"   样本 {i+1}:")
    print(f"     原始分数: {logits_sample.tolist()}")
    print(f"     预测概率: {[f'{p:.3f}' for p in probs_sample.tolist()]}")
    print(f"     预测类别: {class_names[predicted_class]} (置信度: {confidence:.3f})")

print(f"\n7. 不同 dim 值的对比总结：")
print(f"   ┌─────────┬──────────────────┬──────────────────┐")
print(f"   │   dim   │      含义        │   应用场景       │")
print(f"   ├─────────┼──────────────────┼──────────────────┤")
print(f"   │   0     │ 沿批次维度归一化 │ 样本间概率比较   │")
print(f"   │   1     │ 沿类别维度归一化 │ 单样本分类预测   │")
print(f"   │   -1    │ 沿最后一个维度   │ 通用写法         │")
print(f"   └─────────┴──────────────────┴──────────────────┘")

# 温度缩放示例
print(f"\n8. 高级技巧 - 温度缩放 (Temperature Scaling):")
temperatures = [0.5, 1.0, 2.0]
original_logits = torch.tensor([2.0, 1.0, 0.1])
print(f"   原始logits: {original_logits.tolist()}")

for temp in temperatures:
    scaled_logits = original_logits / temp
    probs = nn.Softmax(dim=0)(scaled_logits)
    print(f"   温度 T={temp}: logits={scaled_logits.tolist()}")
    print(f"              概率={[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"              最大概率={probs.max():.3f}")

print(f"\n   温度效果:")
print(f"   - T < 1.0: 增强最大值，使分布更尖锐（更自信）")
print(f"   - T = 1.0: 标准 Softmax")  
print(f"   - T > 1.0: 平滑分布，使预测更保守")

#################################################
# 模型参数
# -------------------------
# 神经网络中的许多层都是 *参数化的*，即具有在训练期间优化的相关权重和偏置。
# 继承 ``nn.Module`` 会自动跟踪模型对象内定义的所有字段，并使所有参数
# 可以通过模型的 ``parameters()`` 或 ``named_parameters()`` 方法访问。
#
# 参数管理的重要性：
# - 训练时需要计算梯度
# - 保存模型时需要存储参数
# - 迁移学习时需要冻结某些参数
# - 正则化需要访问权重
#
# 在这个例子中，我们遍历每个参数，并打印其大小和值的预览。

print(f"模型结构: {model}\n")
print("=" * 60)
print("模型参数详细信息:")
print("=" * 60)

# 遍历所有命名参数
for name, param in model.named_parameters():
    print(f"层名称: {name}")
    print(f"参数形状: {param.size()}")
    print(f"参数数量: {param.numel()}")
    print(f"是否需要梯度: {param.requires_grad}")
    print(f"前几个参数值: {param.flatten()[:5]}")
    print("-" * 40) # 打印40个-

# 计算总参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n模型总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
# 计算模型大小
# *4 是因为float32是4个字节
print(f"模型大小估算: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")

######################################################################
# 总结
# --------------
# 通过这个教程，我们学习了：
# 1. 如何使用 nn.Module 构建神经网络
# 2. 常用层的功能和作用（Linear, ReLU, Flatten, Softmax）
# 3. 使用 Sequential 容器简化网络定义
# 4. 如何管理和查看模型参数
# 5. 神经网络的基本工作原理
#
# 下一步学习建议：
# - 了解损失函数和优化器
# - 学习训练循环的实现
# - 探索更复杂的网络架构（CNN, RNN等）
# - 实践不同的激活函数和正则化技术

#################################################################
# 进一步阅读
# -----------------
# - `torch.nn API 文档 <https://pytorch.org/docs/stable/nn.html>`_
# - `PyTorch 官方教程 <https://pytorch.org/tutorials/>`_
# - `深度学习基础知识 <https://pytorch.org/deep-learning-with-pytorch>`_
