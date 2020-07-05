# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):  #data_file一段文字
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)  # data_lis：句子单词列表 [[a,man,],[],,, ]
        self.data_num = len(self.data_lis)         # data_num：句子数量
        self.indices = range(self.data_num)        # indices : 句子数量的索引
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))   # num_batches :有几个batch   # ceil：向上取整
        self.idx = 0

    def __len__(self):
        return self.num_batches   #几个batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()   #下面函数

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)    #将句子单词列表打乱

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]   #某个batch部分句子索引   [1,2,3,4]   [5,6,7,8]  [9,10,11,12]
        d = [self.data_lis[i] for i in index]                     #某个batch部分句子单词列表   [[a,man,],[],,, ]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)    # [batch_size,1]+[batch_size,单词长度] ,相当于在所有句子单词前加个0，[[0,a,man,],[0,],,, ]
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)  #在所有句子单词后面加0，[[a,man,0],[,,0],,, ]
        self.idx += self.batch_size
        return data, target   #取某个batch的

    def read_file(self, data_file):   # data_file一段文字
        with open(data_file, 'r') as f:
            lines = f.readlines()   # lines所有行
        lis = []
        for line in lines:          # line某一行
            l = line.strip().split(' ')
            l = [int(s) for s in l] # l某个单词 
            lis.append(l)
        return lis     #所有句子单词列表  [[a,man,],[],,, ]

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis      #数据
        self.labels = [1 for _ in range(len(real_data_lis))] +\   #标签
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))   #形成 数据-标签 对
        self.data_num = len(self.pairs)   #句子数量
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self): #同
        return self.num_batches

    def __iter__(self): #同
        return self

    def __next__(self): #同
        return self.next()

    def reset(self): #同
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]   
        label = [p[1] for p in pairs]   
        data = torch.LongTensor(np.asarray(data, dtype='int64'))   #[[a,man,],[],,, ]，前后没加0
        label = torch.LongTensor(np.asarray(label, dtype='int64')) #[1,0,,,]
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):  #同上部分，返回单词列表   #同
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis
