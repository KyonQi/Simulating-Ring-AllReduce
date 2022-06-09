import torch
import torchvision
import numpy as np
import random
from torch import nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0")
NumWorker = 5

class RingWorker(object):
  worker_num = 0
  def __init__(self, net, device, Attack, segmentation):
    torch.manual_seed(12345) #限定随机种子
    torch.cuda.manual_seed(12345)
    self.net = net.to(device)
    self.id = RingWorker.worker_num #每实例化一个worker 就给一个id 0->n
    RingWorker.worker_num += 1
    self.attack = Attack() #Attack是一个攻击类 里面有攻击的成员函数 self.attack是类实例化的对象

    self.segmentation_result = [list(range(j)) for j in segmentation] #segmentation_result放的是分割的层数指标
    for i in range(len(segmentation) - 1): #这个循环完全定义好每一部分负责的Net层数
      for j in range(len(self.segmentation_result[i + 1])): #比如说，16层AlexNet，5个worker
        self.segmentation_result[i + 1][j] += self.segmentation_result[i][-1] + 1 #seg_result就是，[[0,1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]
    
    self.loss_func = nn.CrossEntropyLoss()

    self.remain_gradient = []
    for layer, layer_parameters in enumerate(self.net.parameters()):
      self.remain_gradient.append(torch.zeros_like(layer_parameters).to(DEVICE))
  
  def model_loss(self, data, device): #算训练的损失函数(算参数梯度的) 方便后面更新模型参数 data是train_data
    features, labels = data[0].to(device), data[1].to(device)
    self.net.zero_grad() #梯度清零
    outputs = self.net(features)
    loss = self.loss_func(outputs, labels) #拿到损失函数
    loss.backward() #损失函数求导 


    self.net_gradient = [] #这里把每一层的参数和梯度按照list存 方便后面Send和Gather
    self.net_parameter = [] #这部分跟model_loss计算其实没多大关系 只是为了后面使用方便
    for layer, layer_parameters in enumerate(self.net.parameters()):
      self.net_gradient.append(self.attack.origin((layer_parameters.grad.clone()))) #梯度丢给attack，做攻击处理，之后复制每一层梯度
      self.net_parameter.append(layer_parameters.data.clone()) #复制每一层参数
    
    return loss.item()

  def scatter_send(self, round): #实现Ring AllReduce中 Scatter-Send部分
    with torch.no_grad(): #round从0开始到NumWorker-1
      partition = self.id - round #partition是需要发送的整体部分，是segmentation的部分 也是从0开始
      if partition < 0:
        partition += NumWorker
      if partition >= NumWorker:
        partition -= NumWorker

      partition_gradient = []
      for i in self.segmentation_result[partition]: #segmentation_result[partition]里是具体的层号 0开始
        partition_gradient.append(self.net_gradient[i]) #拿到需要Send的层的梯度list
      return partition_gradient
  
  def scatter_receive(self, round, partition_gradient, attack=False): #实现Scatter-Receive部分 注意还不是AllGather
    with torch.no_grad():                #partition_gradient是send返回的partition_gradient
      partition = self.id - round - 1 #round跟send函数的round是一样的，从0开始到NumWorker-1
      if partition < 0:
        partition += NumWorker #partition是要接受的层的梯度
      if partition >= NumWorker:
        partition -= NumWorker
      
      if attack:
        for i in self.segmentation_result[partition]:
          self.net_gradient[i] = self.net_gradient[i] \
                  + self.attack.inverse(partition_gradient[i - self.segmentation_result[partition][0]]) #攻击梯度汇总
      else:
        for i in self.segmentation_result[partition]:
          self.net_gradient[i] = self.net_gradient[i] \
                    + partition_gradient[i - self.segmentation_result[partition][0]] #梯度汇总
  

      
  def gather_send(self, round): #上面实现了reduce的发、收；这里实现gather的发、收(都是对梯度而言的，不是参数)
    with torch.no_grad(): 
      partition = self.id + 1 - round #与reduce的唯一区别
      if partition < 0:
        partition += NumWorker
      if partition >= NumWorker:
        partition -= NumWorker

      partition_gradient = []
      for i in self.segmentation_result[partition]:
        partition_gradient.append(self.net_gradient[i])
      return partition_gradient
  
  def gather_receive(self, round, partition_gradient):
    with torch.no_grad():
      partition = self.id - round
      if partition < 0:
        partition += NumWorker
      if partition >= NumWorker:
        partition -= NumWorker
      
      for i in self.segmentation_result[partition]:
        self.net_gradient[i] = partition_gradient[i - self.segmentation_result[partition][0]] #self.net_gradient是按层数排的梯度数组
  
  def worker_step(self, lr = LR): #这个函数实现对实例化的worker在Ring AllReduce后实现参数更新
    for segment in self.segmentation_result: #经过scatter/ gather后，net_gradient存有全部累加过的梯度
      for layer in segment:
        self.net_parameter[layer] = self.net_parameter[layer] - lr * self.net_gradient[layer] / (1.0*NumWorker)
    
    for new_param, old_param in zip(self.net_parameter, self.net.parameters()):
      old_param.data = new_param
