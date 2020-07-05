# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):   #main函数中： rollout = Rollout(generator, 0.8)
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, discriminator):  #main函数中： rewards = rollout.get_reward(samples, 16, discriminator)
        """                                       # Monte Carlo？？？
        Args:
            x : (batch_size, seq_len) input data  #数据：单词id
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):  #预测部分单词的句子    #l是前l个单词，不包括第seq_len个单词即最后一个单词在后面的# for the last token ？？？
                data = x[:, 0:l]  
                samples = self.own_model.sample(batch_size, seq_len, data)  #random.sample(list, 5)：从list中随机获取5个元素，作为一个片断返回
                pred = discriminator(samples)  # 鉴别器分数
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)  # i=0，rewards[0]=pred
                else:
                    rewards[l-1] += pred  # i=1, rewards[l-1]=所有单词的reward，

            # for the last token   
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred  #rewards是list，长度seq_len
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):  # main函数中： rollout.update_params()  参数的梯度更新
        dic = {}
        for name, param in self.ori_model.named_parameters():   # 字典{name:param}
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]  # lr*param+(1-lr)*dic[name]
