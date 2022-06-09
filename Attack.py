import torch
import torchvision
import numpy as np
import random
from torch import nn
import matplotlib.pyplot as plt

class Attack(object): #此类实现攻击的成员函数
  #def __init__(self, gradient):
  #  self.origin_gradient = gradient
  
  def origin(self, gradient): #不攻击
    self.origin_gradient = gradient
    return self.origin_gradient
  
  def inverse(self, layer_gradient): #对输入的当前层梯度取反
    self.inverse_gradient = -layer_gradient
    return self.inverse_gradient