import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import *
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "axes.unicode_minus": False #解决负号无法显示的问题
}

import seaborn as sns
colors = sns.color_palette("colorblind", n_colors=6)

marker_size = 8
title_size = 20
xlabel_size = 20
ylabel_size = 20
legend_size = 15

root_path = './smooth'
HMCA = pd.read_csv(os.path.join(root_path,'MAAS_logs_2023-12-12-20-02-39.csv'))
HMCA005 = pd.read_csv(os.path.join(root_path,'MAAS_copy_logs_2023-12-19-13-14-56-lr_0.0005-gamma_0.95-batch_size_256.csv'))
HMCA001 = pd.read_csv(os.path.join(root_path,'MAAS_copy_logs_2023-12-19-15-20-10-lr_0.0001-gamma_0.95-batch_size_256.csv'))

plt.figure(figsize=(6,4))
plt.plot(HMCA005['Step'], HMCA005['Value'], label='LR = 0.0005',marker='^',markersize=marker_size,markerfacecolor='none',color = colors[0])
plt.plot(HMCA['Step'], HMCA['Value'], label='LR = 0.0003',marker='D',markersize=marker_size,markerfacecolor='none',color = colors[1])
plt.plot(HMCA001['Step'], HMCA001['Value'], label='LR = 0.0001',marker='s',markersize=marker_size,markerfacecolor='none',color = colors[2])

plt.xlabel('Episode',fontsize=xlabel_size)
plt.ylabel('Average episode reward',fontsize=ylabel_size)

# 设置横轴记号
xticks(np.linspace(0,2000,5,endpoint=True),fontsize=xlabel_size)
# yticks(np.linspace(-4,4,9,endpoint=True))
yticks(None,fontsize=ylabel_size)

xlim(0,2000)

plt.rcParams['pdf.fonttype'] = 42
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.grid()

if os.path.exists('./convergence') is False:
    os.mkdir('./convergence')


plt.savefig('./convergence/convergence_lr.pdf', bbox_inches='tight')
plt.savefig('./convergence/convergence_lr.eps', bbox_inches='tight')