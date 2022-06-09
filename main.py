import imp
import torch
import torchvision
import numpy as np
import random
from torch import nn
import matplotlib.pyplot as plt
from DataCompose import DataCompose
from Attack import Attack
from RingWorker import RingWorker
from Model import Alexnet, Resnet18
import DrawLossAcc

def ScatterSegmentation(net):
  for i, n in enumerate(net.parameters()):
    #print('layer {0:2d}: size is {1}'.format(int(i+1), n.data.size()))
    continue
  segmentation = [int((i+1) / NumWorker) for j in range(NumWorker)] #segmentation是一个list类型，每个元素表明worker负责的层数
  for j in range(i+1 - int((i+1) / NumWorker) * NumWorker):
    segmentation[j] += 1 #倘若不能整数切割 需要这个
  print("Net Segmentation is:{0}".format(segmentation))
  return segmentation

torch.manual_seed(12345)
torch.cuda.manual_seed(12345)
random.seed(12345)
np.random.seed(12345)

NumAttacker = 1
Iteration = 2000
Show_Iter = 10
TrainBatchSize = 128
TestBatchSize = 16
net = Alexnet() # net = Resnet18()
NumWorker = 5
NumAttacker = 1
DEVICE = torch.device("cuda:0")
LR = 0.25

data_compose = DataCompose() #实例化取数据类
segmentation = ScatterSegmentation(net)

worker_list = []
attacker_list = [1 for _ in range(NumAttacker)] + [0 for _ in range(NumWorker - NumAttacker)] 
#attacker_list #如果是攻击者，则标记为1
loss_list = []
accu_list = []

for i in range(NumWorker):
  worker_list.append(RingWorker(net, DEVICE, Attack, segmentation)) #依次实例化worker

total_loss = 0.0

for iter in range(Iteration): #训练过程
  iter_loss = 0.0
  for i in range(NumWorker):
    worker_loss = worker_list[i].model_loss(data_compose.step(TrainBatchSize), DEVICE)
    iter_loss += worker_loss
  iter_loss /= NumWorker
  total_loss += iter_loss

  for scatter_round in range(NumWorker-1):
    for i in range(NumWorker): #i是第i个发送梯度的worker
      i_receive = i + 1 #i_receive是接受梯度的worker
      if i_receive >= NumWorker:
        i_receive -= NumWorker
      
      if attacker_list[i_receive] == 1:
        worker_list[i_receive].scatter_receive(scatter_round, worker_list[i].scatter_send(scatter_round), attack=False)
      else:
        worker_list[i_receive].scatter_receive(scatter_round, worker_list[i].scatter_send(scatter_round), attack=False)
  
  for gather_round in range(NumWorker-1):
    for i in range(NumWorker): #i是第i个发送完整梯度的worker
      i_receive = i + 1
      if i_receive >= NumWorker:
        i_receive -= NumWorker
      worker_list[i_receive].gather_receive(gather_round, worker_list[i].gather_send(gather_round))
  
  for i in range(NumWorker): #所有worker更新参数
    worker_list[i].worker_step(LR)
  
  if (iter+1) % Show_Iter == 0:
    #print(total_loss / Show_Iter)
    loss_list.append(total_loss / Show_Iter) #损失
    total_loss = 0.0
    accu_list.append(data_compose.model_test(worker_list[0].net, DEVICE)) #精度

Loss_list = loss_list  #存储每次epoch损失值
Accu_list = accu_list
DrawLossAcc.draw_loss(Loss_list, int(Iteration / Show_Iter))
DrawLossAcc.draw_acc(Accu_list, int(Iteration / Show_Iter))