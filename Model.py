import torch
import torchvision
import numpy as np
import random
from torch import nn
import matplotlib.pyplot as plt

# 实现Alexnet和Resnet18的搭建
class Alexnet(nn.Module): #实现Alexnet
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU() 
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    self.dense = nn.Sequential(
        nn.Flatten(), #展平给全连接
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
  
  def forward(self, X):
    X = self.conv1(X)
    X = self.conv2(X)
    X = self.conv3(X)
    X = self.conv4(X)
    X = self.conv5(X)
    X = self.dense(X)
    return X

#实现Resnet18
class Residual(nn.Module): #实现残差块
  def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(output_channels)
    self.bn2 = nn.BatchNorm2d(output_channels)
  
  def forward(self, X):
    Y = nn.functional.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    Y = nn.functional.relu(Y)
    return Y

def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
  blk = []
  for i in range(num_residuals): #num_residuals为需要的残差块数
    if i == 0 and not first_block:
      blk.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2))
    else:
      blk.append(Residual(output_channels, output_channels))
  return blk

class Resnet18(nn.Module):
  def __init__(self):
    super().__init__()
    self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
    self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
    self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
    self.dense = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))

  def forward(self, X):
    X = self.b1(X)
    X = self.b2(X)
    X = self.b3(X)
    X = self.b4(X)
    X = self.b5(X)
    X = self.dense(X)
    return X
