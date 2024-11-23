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

y1=[0.8013, 0.8013]


y2=[0.8316, 0.8316]


y3=[0.8256, 0.8256]

y4=[0.8186, 0.8186]

y5=[0.7823, 0.8086, 0.8127, 0.8204, 0.8255, 0.8286, 0.8316]

if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = [u'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.size'] = 14
    row_x = [0, 5]
    plt.plot(row_x, y1, label='FedSage+', ls='-.', c='#548A5A', linewidth=3)



    row_x = [0, 5]
    plt.plot(row_x, y3, label='FedGCN (2-hop)', ls='-.', c='#A5E49E', linewidth=3)


    row_x = [0, 5]
    plt.plot(row_x, y4, label='FedCog', ls='-.', c='#4BCA3E', linewidth=3)

    row_x = [0, 5]
    plt.plot(row_x, y2, label='FedHGL with HC', ls='--', c='grey', linewidth=3)

    row_x = [1, 1.5, 1.8, 2, 2.5, 3, 5]
    plt.plot(row_x, y5, label='LDP-FedHGL', ls='-', c='#1f77b4', marker='s', linewidth=3)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.ylim(0.75, 0.85)
    plt.xlim(1, 5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    plt.xlabel('Epsilon',fontdict={'fontsize': 32, 'fontname': 'Times New Roman'})
    # plt.ylabel('故障分类准确率(%)')
    # plt.xlabel('训练迭代周期')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.rcParams['savefig.dpi'] = 300 # Image pixels
    plt.show()
