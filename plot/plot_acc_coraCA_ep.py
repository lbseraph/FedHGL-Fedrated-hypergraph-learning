import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

config = {
    "font.family":'Times New Roman',  # Set the font type
    "axes.unicode_minus": False, # Solve the problem that the negative sign cannot be displayed
    'font.size': 20
}
rcParams.update(config)

y1=[0.4833, 0.4833]


y2=[0.6778, 0.6778]


y3=[0.4015, 0.4015]

y4=[0.5044, 0.5044]

y5=[0.3335, 0.4233, 0.4819, 0.5521, 0.6006, 0.6204, 0.6462]

if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = [u'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.size'] = 14
    row_x = [0, 2]
    plt.plot(row_x, y1, label='FedHGL w/o HC', ls='-.', c='#5977D5', linewidth=3)



    row_x = [0, 2]
    plt.plot(row_x, y3, label='Federated HNHN', ls='-.', c='#83B5FF', linewidth=3)


    row_x = [0, 2]
    plt.plot(row_x, y4, label='Federated HyperGCN', ls='-.', c='#165584', linewidth=3)

    row_x = [0, 2]
    plt.plot(row_x, y2, label='FedHGL with HC', ls='--', c='grey', linewidth=3)

    row_x = [0.1, 0.2, 0.3, 0.5, 0.8, 1, 2]
    plt.plot(row_x, y5, label='LDP-FedHGL', ls='-', c='#1f77b4', marker='s', linewidth=3)


    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.ylim(0, 0.8)
    plt.xlim(0, 2)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    plt.xlabel('Epsilon',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    # plt.ylabel('故障分类准确率(%)')
    # plt.xlabel('训练迭代周期')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.rcParams['savefig.dpi'] = 300 # Image pixels
    plt.show()
