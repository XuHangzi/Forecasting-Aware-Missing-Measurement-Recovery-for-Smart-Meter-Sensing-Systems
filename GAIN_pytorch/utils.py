# coding=utf-8
#
# Apache License 2.0
# 可免费使用本代码，但需遵守许可证条款
#

'''GAIN 模型 PyTorch 版本工具函数。

包含以下功能:
(1) normalization: MinMax 归一化
(2) renormalization: 从归一化数据恢复原始数据
(3) rounding: 对分类变量进行取整
(4) rmse_loss: 计算 RMSE 评估填充效果
(5) binary_sampler: 采样二进制随机变量
(6) uniform_sampler: 采样均匀分布随机变量
(7) sample_batch_index: 采样小批量数据的索引
'''

import numpy as np


def normalization(data, parameters=None):
    '''将数据归一化到 [0, 1] 范围。
    
    参数:
        - data: 原始数据
    
    返回:
        - norm_data: 归一化后的数据
        - norm_parameters: 归一化参数（每列的最小值和最大值）
    '''
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
        # MinMax 归一化
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        # 对每一列进行处理
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)
        
        # 返回归一化参数用于反归一化
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
        # 对每一列进行处理
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
        
        norm_parameters = parameters
    
    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''将归一化数据从 [0, 1] 范围恢复到原始范围。
    
    参数:
        - norm_data: 归一化后的数据
        - norm_parameters: 归一化参数
    
    返回:
        - renorm_data: 恢复到原始范围的数据
    '''
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]
    
    return renorm_data


def rounding(imputed_data, data_x):
    '''对分类变量进行取整处理。
    
    参数:
        - imputed_data: 填充后的数据
        - data_x: 原始数据（包含缺失值）
    
    返回:
        - rounded_data: 取整后的填充数据
    '''
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # 仅对分类变量进行处理（唯一值数量小于20）
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    
    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    '''计算原始数据与填充数据之间的 RMSE。
    
    参数:
        - ori_data: 原始完整数据（无缺失）
        - imputed_data: 填充后的数据
        - data_m: 缺失指示矩阵
    
    返回:
        - rmse: 均方根误差
    '''
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    
    # 仅计算缺失值位置的误差
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)
    
    rmse = np.sqrt(nominator / float(denominator))
    
    return rmse


def binary_sampler(p, rows, cols):
    '''采样二进制随机变量。
    
    参数:
        - p: 取值为1的概率
        - rows: 行数
        - cols: 列数
    
    返回:
        - binary_random_matrix: 生成的二进制随机矩阵
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    '''采样均匀分布随机变量。
    
    参数:
        - low: 下界
        - high: 上界
        - rows: 行数
        - cols: 列数
    
    返回:
        - uniform_random_matrix: 生成的均匀分布随机矩阵
    '''
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size):
    '''采样小批量数据的索引。
    
    参数:
        - total: 总样本数
        - batch_size: 批量大小
    
    返回:
        - batch_idx: 批量索引
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx