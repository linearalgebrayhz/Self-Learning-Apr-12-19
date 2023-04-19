import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration: 指定使用GPU还是CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters: 设置超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset: 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader: 构造数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
# 定义一个全连接神经网络，包含一个隐藏层
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # input_size -> hidden_size
        self.relu = nn.ReLU() # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes) # hidden_size -> num_classes
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out) #全连接层
        return out

# 实例化神经网络
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer: 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model: 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device: 将张量移动到指定设备上
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass: 前向传播计算输出
        outputs = model(images)
        loss = criterion(outputs, labels) # 计算损失
        
        # Backward and optimize: 反向传播，更新参数
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        # Print loss every 100 steps: 每100步打印一次损失值
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model: 测试模型
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint: 保存模型参数
torch.save(model.state_dict(), 'model.ckpt')
