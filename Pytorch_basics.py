import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward() # 反向求导，详见梯度计算

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

print(x)
print(y)

# Build a fully connected layer.
linear = nn.Linear(3, 2) # create linear layer, randomize a 2 by 3 weight mat and 2 by 1 bias vector for training purpose
print ('w: ', linear.weight)
print ('b: ', linear.bias)
# 疑问1：y=wx+b 是怎么乘出来的？

'''
w:  Parameter containing:
tensor([[ 0.0743,  0.2708,  0.2810],
        [ 0.0479,  0.2777, -0.0240]], requires_grad=True)
b:  Parameter containing:
tensor([-0.1443, -0.0450], requires_grad=True)
'''
# Build loss function and optimizer.
criterion = nn.MSELoss() 
# 损失函数：计算误差均方
# 类似于求方差

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) 
#优化器
# This is a Stochastic Gradient Descent optimizer
# linear.parameters() -> parameters of linear: weight matrix and bias vector
# lr=0.01 -> the step of training: 学习率

'''
在训练模型时，通常需要将输入数据和真实值转换为 PyTorch 的张量格式，并将其传递给模型进行前向传播计算。
然后，计算损失函数的值，并使用反向传播算法自动计算损失函数对模型参数的导数，更新模型参数。这个过程可以通过以下代码实现：

# 前向传播
pred = linear(inputs)

# 计算损失函数
loss = criterion(pred, labels)

# 反向传播
loss.backward()

# 更新模型参数
optimizer.step()

# 清空梯度
optimizer.zero_grad()
其中，inputs 表示输入数据的张量格式，labels 表示真实值的张量格式，pred 表示模型的预测值的张量格式。
loss.backward() 函数表示反向传播，自动计算损失函数对模型参数的导数，并将其存储在对应的张量的 .grad 属性中。
optimizer.step() 函数表示使用优化器更新模型参数，使得损失函数的值最小化。optimizer.zero_grad() 函数表示清空梯度，以便进行下一次迭代。
'''
# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
# 疑问2：反向传播，每一次权重矩阵和偏差向量都会有所不同？

# 1-step gradient descent.
optimizer.step()
# 作用原理： 如，对于随机梯度下降法，更新公式为 -> parameters = parameters - learning_rate(lr) * grad
# 对于权重矩阵 W 的每个元素 w_{i,j}，都会被同时调整，即减去学习率乘以相应的梯度
# 疑问3：？具体运算方式，回顾梯度和矩阵


# You can also perform gradient descent at the low level.
linear.weight.data.sub_(0.01 * linear.weight.grad.data)
linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
print('loss after 2 step optimization: ', loss.item())
# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
print('loss after 3 step optimization: ', loss.item())
# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
print('loss after 4 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
print(y)
# Convert the torch tensor to a numpy array.
z = y.numpy()
print(z)

# ================================================================== #
#                         4. Input pipeline                          #
# ================================================================== #
Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/', # 数据集存放的根目录
                                             train=True, # 加载训练集or测试集        
                                             transform=transforms.ToTensor(), # transform预处理，将图片转化为张量
                                             download=True)

'''
CIFAR10 是一个经典的图像分类数据集，包含了 10 个类别的 RGB 彩色图像，每个类别有 6,000 张训练图像和 1,000 张测试图像。
在 PyTorch 中，可以使用 torchvision.datasets.CIFAR10 类来加载 CIFAR10 数据集。
加载数据集后，可以使用 PyTorch 中的 DataLoader 类来对数据进行批量处理，以便进行训练和测试。
'''

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # 要训练的数据集
                                           batch_size=64, 
                                           shuffle=True) # 是否重排
'''
DataLoader 是 PyTorch 中用于加载数据的类，它可以对数据进行批量处理，支持多线程和异步预加载等功能。
使用 DataLoader 可以将数据集分成多个 batch，每个 batch 中包含多个样本，方便进行训练。
'''
# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader) 
# 将DataLoader 对象 train_loader 转换为一个迭代器 data_iter

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

for batch_idx, (images, labels) in enumerate(train_loader):
    # 训练代码
    pass

'''
具体来说，images 的数据类型是 torch.Tensor，它通常是一个四维张量，表示一个 batch 中的图像数据。
例如，如果一个 batch 中有 64 张图像，每张图像的大小是 3x32x32（通道数为 3，高和宽均为 32），那么 images 的形状应该是 (64, 3, 32, 32)。

labels 的数据类型通常是 torch.Tensor 或者 torch.LongTensor，它通常是一个一维张量，表示一个 batch 中每张图像的标签。
例如，如果一个 batch 中有 64 张图像，那么 labels 的形状应该是 (64,)。
'''

'''
# 如何理解pipeline

通常情况下，机器学习的 Pipeline 包括以下几个步骤：

数据预处理：包括数据清洗、特征提取、特征选择等操作，以便将原始数据转换为有效的特征。

模型训练：选择合适的模型，并使用训练数据对模型进行训练，从而得到一个能够对新数据进行预测的模型。

模型评估：使用测试数据对模型进行评估，以便了解模型的泛化能力和性能表现。

模型部署：将训练好的模型部署到生产环境中，以便对新数据进行预测和处理。
'''

# ================================================================== #
#                5. Input pipeline for custom dataset                #
# ================================================================== #

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# 需要一定的python OOP的基础
# 以下是一个根据以上模板构建的dataset示例
import os
import pandas as pd
from PIL import Image

class CustomDataset(torch.utils.data.Dataset): # CustomDataset 继承自 torch.utils.data.Dataset 类
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # 读取 csv 文件中的图像文件名和标签信息
        df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
        for _, row in df.iterrows():
            filename = row['filename']
            label = row['label']
            self.image_paths.append(os.path.join(data_dir, 'train_data', filename))
            self.labels.append(label)

    def __getitem__(self, index):
        # 读取图像文件，并转换为 RGB 模式
        image = Image.open(self.image_paths[index]).convert('RGB')
        # 对图像进行预处理
        if self.transform is not None:
            image = self.transform(image)
        # 返回图像和标签
        return image, self.labels[index]

    def __len__(self):
        # 返回数据集的大小
        return len(self.labels)

# 构建数据集
custom_dataset = CustomDataset(data_dir='path/to/data', transform=None)
# 构建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)
# ResNet-18 is a classic pretrained Deep Convolutional Neural Network

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False
'''
即将ResNet-18中所有参数冻结起来，不参与反向传播更新梯度。
这样可以避免在对整个模型进行微调时，对预训练的部分参数产生过大的改变
fine-tune
'''

# Replace the top layer for finetuning.
# 这里建立全新的一层
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example. 类别数量
# resnet.fc.in_features = 512
# resnet.fc 是指全连接层，输入是512维向量

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

'''
补充了解：ResNet-18 架构
输入层：ResNet-18的输入是一张 224 * 224 的RGB图片。
卷积层：ResNet-18包含了一系列的卷积层，以提取输入图片的特征。这些卷积层包括一个 $7 \times 7$ 的卷积层，跟随着四个包含了多个 $3 \times 3$ 卷积层的块。在每个块之间，还包括了一个 $1 \times 1$ 的卷积层来调整输出特征通道的数目。
残差块：ResNet-18通过残差边来构建一个深度的卷积神经网络。每个残差块都由两个卷积层和一个跨越恒等映射的残差边组成。这些残差边使得网络可以学习恒等映射，从而避免了梯度消失和梯度爆炸问题。
全局平均池化层：在ResNet-18的末尾，通过一个全局平均池化层将最后一个残差块的输出压缩为一个特征向量。这个特征向量可以被用于图像分类任务。
全连接层：最后一个全连接层将特征向量映射到ImageNet数据集的1000个类别上，用于分类任务。
'''
# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

'''
torch.save(resnet, 'model.ckpt') 语句将 PyTorch 模型 resnet 保存到文件 'model.ckpt' 中。
这个文件可以包含模型的所有参数和状态信息，以及一些元数据（例如模型的架构和版本信息）。
在训练模型后，我们可以使用这个文件来保存模型，以便在以后的时间内重新加载模型，而无需重新训练模型。

torch.load('model.ckpt') 语句从文件 'model.ckpt' 中加载模型。
这个函数返回一个 torch.nn.Module 对象，它包含从文件中加载的模型的所有参数和状态信息。
我们可以使用这个对象来进行预测、评估或微调模型。


'''
# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
'''
这种保存方式通常更加常见，因为它只包含模型的参数，而不是整个模型对象。
这使得我们可以更加灵活地加载和保存模型参数，例如在不同的计算机、设备或平台之间共享模型参数。

resnet.load_state_dict(torch.load('params.ckpt')) 语句从文件 'params.ckpt' 中加载模型参数，并将它们设置为 resnet 模型对象的参数。
这个方法只需加载模型参数，而不需要重新构建整个模型，因此它比 torch.load() 方法更加轻量级。
请注意，我们必须首先创建一个与原始模型结构相同的模型对象，然后再加载参数。
这是因为模型参数必须与模型结构相匹配，否则加载参数将失败。

请注意，与 torch.load() 方法不同，load_state_dict() 方法只返回模型的参数，而不是整个模型对象。
因此，我们必须首先创建一个模型对象，然后将参数加载到该对象中。
这意味着我们需要知道模型的架构和版本信息，以便正确地创建模型对象。
'''

