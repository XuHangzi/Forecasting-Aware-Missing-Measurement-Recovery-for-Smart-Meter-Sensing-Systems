# coding=utf-8
#
# Apache License 2.0
# 可免费使用本代码，但需遵守许可证条款
#

'''GAIN 模型 PyTorch 版本实现。

GAIN (Generative Adversarial Imputation Networks) 用于缺失数据填充的生成对抗网络。
'''

import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    '''生成器网络。
    
    将拼接后的 [data, mask] 作为输入，输出填充值。
    '''
    
    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.dim = dim
        self.h_dim = h_dim
        
        # 生成器层结构
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, m):
        '''前向传播。
        
        参数:
            - x: 输入数据（缺失值位置用随机噪声替代）
            - m: 掩码向量（1=观测值，0=缺失值）
            
        返回:
            - G_prob: 生成的填充值，范围 [0, 1]（sigmoid 输出）
        '''
        # 拼接掩码和数据
        inputs = torch.cat([x, m], dim=1)
        
        G_h1 = self.relu(self.fc1(inputs))
        G_h2 = self.relu(self.fc2(G_h1))
        G_prob = self.sigmoid(self.fc3(G_h2))
        
        return G_prob


class Discriminator(nn.Module):
    '''判别器网络。
    
    将拼接后的 [combined_data, hint] 作为输入，输出概率。
    '''
    
    def __init__(self, dim, h_dim):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.h_dim = h_dim
        
        # 判别器层结构
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h):
        '''前向传播。
        
        参数:
            - x: 组合数据（观测值 + 填充值）
            - h: 提示向量
            
        返回:
            - D_prob: 判别为"真实观测值"的概率
        '''
        # 拼接数据和提示
        inputs = torch.cat([x, h], dim=1)
        
        D_h1 = self.relu(self.fc1(inputs))
        D_h2 = self.relu(self.fc2(D_h1))
        D_logit = self.fc3(D_h2)
        D_prob = self.sigmoid(D_logit)
        
        return D_prob


def xavier_init_(layer):
    '''Xavier 权重初始化。'''
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)