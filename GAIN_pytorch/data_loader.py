# coding=utf-8
#
# Apache License 2.0
# 可免费使用本代码，但需遵守许可证条款
#

'''数据加载器，用于加载 UCI 数据集（letter, spam）和电量数据集。
'''

import os
import numpy as np
from utils import binary_sampler


def data_loader(data_name, miss_rate):
    '''加载数据集并引入缺失值。
    
    参数:
        - data_name: 数据集名称（letter, spam, 或电量数据集名称）
        - miss_rate: 缺失值的概率
    
    返回:
        data_x: 原始完整数据
        miss_data_x: 引入缺失值后的数据
        data_m: 缺失值指示矩阵（1=观测值，0=缺失值）
    '''
    # 获取当前脚本所在目录，确保路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # 加载数据
    if data_name in ['letter', 'spam', 'LD2011_2014', 'LD2011_2014_train', 'LD2011_2014_test',
                     'electricity_data_2013_processed', 'electricity_data_2014_processed']:
        file_name = os.path.join(data_dir, data_name + '.csv')
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    
    # 数据维度
    no, dim = data_x.shape
    
    # 引入缺失值
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan
    
    return data_x, miss_data_x, data_m