# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function

    Args:
        weight: Tensor (num_class, )
    """
    def __init__(self, weight):
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):   #这两者之间的loss计算方式
        """
        Args:
            prob: (N, C)    # N行C列的矩阵
            target : (N, )  # N行一列
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))   # [1,C]  还是[N,1],应该前者
        weight = weight.expand(N, C)  # (N, C)
        if prob.is_cuda:
            weight = weight.cuda()
        prob = weight * prob     # (N, C)*(N, C)？？？

        one_hot = torch.zeros((N, C))  # (N, C)全0
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        return -torch.sum(loss)  # loss矩阵内的每个词相加



