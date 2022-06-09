import torch
import torchvision
import numpy as np
import random
from torch import nn
import matplotlib.pyplot as plt

TrainBatchSize = 128
TestBatchSize = 16

class DataCompose():
  def __init__(self):
    train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    self.train_set = torchvision.datasets.CIFAR10(root='./data/train', train=True, download=True, transform=train_augs)
    self.test_set = torchvision.datasets.CIFAR10(root='./data/test', train=False, download=True, transform=test_augs)

    self.train_data = torch.utils.data.DataLoader(self.train_set, batch_size=TrainBatchSize, shuffle=True) #这个暂时没用到 不知道用next能不能实现step需要的效果
    self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size=TestBatchSize, shuffle=False, drop_last=True)

    self.flag = 0
    self.flag_list = np.arange(50000)
    self.accuracy_list = []
    np.random.shuffle(self.flag_list) #随机打乱取图片 这样能不能保证IID暂时不确定 后面重新写个IID的函数
  
  def step(self, batch_size=TrainBatchSize): #这个函数每次给一个BatchSize大小的数据用于求损失
    features = []                            #这个函数后续有时间会被一个IID分割的函数代替 有时间再重写
    labels = []
    for _ in range(batch_size):
      feature, label = self.train_set[self.flag_list[self.flag]]
      features.append(feature)
      labels.append(label)
      self.IfAll()
    features = torch.stack(features, 0)
    labels = torch.tensor(labels)
    return (features, labels)
  
  def IfAll(self):
    if (self.flag+1) >= len(self.train_set):
      self.flag = 0
    else:
      self.flag += 1
    if self.flag == 0:
      np.random.shuffle(self.flag_list)

  def model_test(self, net, device=torch.device("cuda:0")):
    correct = 0
    total_img = 0
    with torch.no_grad():
      net.eval()
      for features, labels in self.test_data:
        features, labels = features.to(device), labels.to(device) #拿的一个batch_size
        outputs = net(features)
        _, indexes = torch.max(outputs, 1) #torch.max返回两个值 第一个是值 第二个是索引
        correct += (labels == indexes).squeeze().sum().item()
        total_img += len(labels)
      
      accuracy = (1.0 * correct) / total_img
      net.train()
    return accuracy
