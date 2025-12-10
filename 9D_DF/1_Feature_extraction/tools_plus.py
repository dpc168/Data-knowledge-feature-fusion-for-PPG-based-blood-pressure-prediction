import math
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from pylab import mpl
import pandas as pd
# mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import openpyxl


# 0-1归一化
def min_max_normalization(data):
    result = list()
    min_ = min(data)
    max_ = max(data)
    for i in range(0, len(data)):
        result.append((data[i] - min_) / (max_ - min_))
    return result


# 计算PSQI
def calculate_perfusion_sqi(raw_signal, filtered_signal):
    """
    计算PPG信号的灌注指数(P_SQI) - 改进版

    参数:
    - raw_signal: 原始PPG信号(一维numpy数组)
    - filtered_signal: 预处理后的PPG信号(一维numpy数组)

    返回:
    - p_sqi: 灌注指数值

    公式:
    P_SQI = [(y_max - y_min) / |x_mean|] × 100
    其中:
    - y是滤波后的PPG信号(filtered_signal)
    - x是原始PPG信号(raw_signal)
    - x_mean是原始信号的平均值
    """

    # 计算滤波后信号的峰峰值
    y_peak_to_peak = np.max(filtered_signal) - np.min(filtered_signal)

    # 计算原始信号绝对值的均值(加小值防止除零)
    x_mean_abs = np.abs(np.mean(raw_signal)) + 1e-6

    # 计算灌注指数
    p_sqi = (y_peak_to_peak / x_mean_abs) * 100

    return p_sqi

# 读取信号
def Read_raw_ppg_signal(file_path):
    """
    :param file_path:读取文件路径
    :return:返回数据
    """
    f = open(file_path, 'r', encoding='utf-8')
    str = f.readlines()
    str = str[0]
    str = str.strip()
    str = str.split('\t')
    data = []
    for i in range(0, 2100):
        data.append(int(float(str[i])))
    return data

# 去基线
def detrend(data):
    x_1 = 1
    x_2 = len(data)
    y_1 = data[0]
    y_2 = data[-1]
    y = list()
    x = np.linspace(1, len(data), len(data))
    for i in range(0, len(data)):
        y.append(y_1 + ((y_1 - y_2) * (x[i] - x_1)) / (x_1 - x_2))
    result = np.round(np.subtract(data, y).astype(float))
    return result

# 2.寻找局部最小值
def local_min(data, window_size):
    if window_size % 2 == 0 or window_size <= 0:  # 判断错误情况
        print("第二个参数请输入奇数且大于0!")
        return None
    else:
        min_array = []  # 保存局部最小值
        min_index_array = []  # 保存局部最小值下标
        data_length = len(data)  # 数据的长度
        start = 0  # 窗口的起始位置
        end = window_size - 1  # 窗口的结束位置
        p = int((start + end) / 2)  # 窗口中间的位置

        if data_length < window_size:  # 如果出现窗口大小比数据大时，返回None
            return None
        else:
            while end < data_length:  # 窗口开始滑动

                temp_left = 0  # 标记
                for i in range(0, math.floor(window_size / 2)):  # 先判断左边
                    if data[p] < data[start + i]:
                        temp_left = temp_left + 1  # 满足条件标记+1
                        continue
                    else:
                        break

                temp_right = 0  # 标记
                for i in range(0, math.floor(window_size / 2)):  # 再判断右边
                    if data[p] < data[p + i + 1]:
                        temp_right = temp_right + 1  # 满足条件标记+1
                        continue
                    else:
                        break

                # 窗口开始滑动
                if temp_right == math.floor(window_size / 2) and temp_left == math.floor(
                        window_size / 2):  # 看左右2边标记是否相等
                    min_array.append(data[p])  # 将局部最小值加入min_array
                    min_index_array.append(p)  # 将局部最小值加入min_index_array
                    start = start + 1  # 更新窗口的起始位置
                    end = end + 1  # 更新窗口的结束位置
                    p = int((start + end) / 2)  # 更新窗口的中间位置
                else:  # 左右2边标记不相等
                    start = start + 1  # 更新窗口的起始位置
                    end = end + 1  # 更新窗口的结束位置
                    p = int((start + end) / 2)  # 更新窗口的中间位置
                # 开始下一次检测
            return min_array, min_index_array  # 返回最小值和最小值下标


# 3.寻找局部最大值
# 起点从下标0开始
def local_max(data, window_size):
    if window_size % 2 == 0 or window_size <= 0:  # 判断错误情况
        print("第二个参数请输入奇数且大于0!")
        return None
    else:
        max_array = []  # 保存局部最大值
        max_index_array = []  # 保存局部最大值下标

        data_length = len(data)  # 数据的长度
        start = 0  # 窗口的起始位置
        end = window_size - 1  # 窗口的结束位置
        p = int((start + end) / 2)  # 窗口中间的位置

        if data_length < window_size:  # 如果出现窗口大小比数据大时，返回None
            return None
        else:
            while end < data_length:  # 窗口开始滑动

                temp_left = 0  # 标记
                for i in range(0, math.floor(window_size / 2)):  # 先判断左边
                    if data[p] > data[start + i]:
                        temp_left = temp_left + 1  # 满足条件标记+1
                        continue
                    else:
                        break

                temp_right = 0  # 标记
                for i in range(0, math.floor(window_size / 2)):  # 再判断右边
                    if data[p] > data[p + i + 1]:
                        temp_right = temp_right + 1  # 满足条件标记+1
                        continue
                    else:
                        break

                # 窗口开始滑动
                if temp_right == math.floor(window_size / 2) and temp_left == math.floor(
                        window_size / 2):  # 看左右2边标记是否相等
                    max_array.append(data[p])  # 将局部最大值加入max_array
                    max_index_array.append(p)
                    start = start + 1  # 更新窗口的起始位置
                    end = end + 1  # 更新窗口的结束位置
                    p = int((start + end) / 2)  # 更新窗口的中间位置
                else:  # 左右2边标记不相等
                    start = start + 1  # 更新窗口的起始位置
                    end = end + 1  # 更新窗口的结束位置
                    p = int((start + end) / 2)  # 更新窗口的中间位置
                # 开始下一次检测
            return max_array, max_index_array


# 插值特征
def ppg_width(data, n):
    width_index = list()
    width_value = list()
    y_value = list()
    # data = data.tolist()
    peak_value = max(data)
    peak_index = data.index(peak_value)

    weight = peak_value / n
    for i in range(0, n - 1):
        y_value.append(weight * (i + 1))

    for i in range(0, len(y_value)):
        for j in range(0, len(data)):
            if data[j] >= y_value[i]:
                width_value.append(data[j])
                width_index.append(j)
                break
    flag = -1
    for i in range(0, len(y_value)):
        for j in range(peak_index, len(data)):
            if data[j] <= y_value[flag]:
                width_value.append(data[j])
                width_index.append(j)
                flag = flag - 1
                break
    width_index = np.add(width_index, 1)
    start = 1
    start_value = data[start - 1]
    end = len(data)
    end_value = data[end - 1]
    width_index = width_index.tolist()
    width_index.insert(0, start)
    width_value.insert(0, start_value)
    width_index.append(end)
    width_value.append(end_value)
    width_index.insert(int(len(width_index) / 2), peak_index + 1)
    width_value.insert(int(len(width_value) / 2), peak_value)
    return width_index, width_value