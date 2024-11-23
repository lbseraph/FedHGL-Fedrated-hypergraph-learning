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

y1=[0.7755, 0.7755]


y2=[0.8396, 0.8396]


y3=[0.8151, 0.8151]

y4=[0.8171, 0.8171]

y5=[0.7471, 0.8016, 0.8064, 0.8115, 0.8163, 0.8188, 0.8257, 0.8322]

if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = [u'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.size'] = 14
    row_x = [0, 6]
    plt.plot(row_x, y1, label='FedSage+', ls='-.', c='#548A5A', linewidth=3)



    row_x = [0, 6]
    plt.plot(row_x, y3, label='FedGCN (2-hop)', ls='-.', c='#A5E49E', linewidth=3)


    row_x = [0, 6]
    plt.plot(row_x, y4, label='FedCog', ls='-.', c='#4BCA3E', linewidth=3)

    row_x = [0, 6]
    plt.plot(row_x, y2, label='FedHGL with HC', ls='--', c='grey', linewidth=3)

    row_x = [1, 2, 2.2, 2.5, 2.8, 3, 4, 6]
    plt.plot(row_x, y5, label='LDP-FedHGL', ls='-', c='#1f77b4', marker='s', linewidth=3)

    plt.ylim(0.7, 0.85)
    plt.xlim(1, 6)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    plt.xlabel('Epsilon',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    # plt.ylabel('故障分类准确率(%)')
    # plt.xlabel('训练迭代周期')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.rcParams['savefig.dpi'] = 300 # Image pixels
    plt.show()
