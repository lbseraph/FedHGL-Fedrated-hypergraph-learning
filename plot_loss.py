import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False, #解决负号无法显示的问题
    'font.size': 14
}
rcParams.update(config)


if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = [u'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.size'] = 14
    row_x = []
    for i in range(len(y1)):
        row_x.append(i)
    plt.plot(row_x, y1, label='Global HGNN', ls='-')

    row_x = []
    for i in range(len(y2)):
        row_x.append(i)
    plt.plot(row_x, y2, label='Local HGNN', ls='-.')

    row_x = []
    for i in range(len(y3)):
        row_x.append(i)
    plt.plot(row_x, y3, label='FedHGN w/o HC', ls=':')

    row_x = []
    for i in range(len(y4)):
        row_x.append(i)
    plt.plot(row_x, y4, label='FedHGN with HC', ls='--')

    plt.ylim(0, 2)
    plt.xlim(0, 100)
    plt.legend(loc='lower right')
    plt.ylabel('Train Loss')
    plt.xlabel('Communication Rounds')
    # plt.ylabel('故障分类准确率(%)')
    # plt.xlabel('训练迭代周期')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.rcParams['savefig.dpi'] = 300 # 图片像素
    plt.show()
