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
rcParams.update(config)

import seaborn as sns
colors = sns.color_palette("colorblind", n_colors=6)

marker_size = 8
title_size = 20
xlabel_size = 20
ylabel_size = 18
legend_size = 12

root_path = './smooth'
HMCA = pd.read_csv(os.path.join(root_path,'MAAS_logs_2023-12-12-20-02-39.csv'))
SAC = pd.read_csv(os.path.join(root_path,'SAC_logs_2023-12-12-15-36-57.csv'))
DDPG = pd.read_csv(os.path.join(root_path,'DDPG_logs_2023-12-12-15-40-14.csv'))
SAL = pd.read_csv(os.path.join(root_path,'SAL_logs_2023-12-12-22-55-40.csv'))
PPO = pd.read_csv(os.path.join(root_path,'PPO_results_MyEnv_MyEnv_mappo_check_run1_logs.csv'))

plt.figure(figsize=(4.5,3))

plt.plot(HMCA['Step'], HMCA['Value'], label='HMCA',marker='^',markersize=marker_size,markerfacecolor='none',color = colors[0])
plt.plot(PPO['Step'], PPO['Value'], label='MAPPO',marker='o',markersize=marker_size,markerfacecolor='none',color = colors[1])
plt.plot(SAC['Step'], SAC['Value'], label='MASAC',marker='D',markersize=marker_size,markerfacecolor='none',color = colors[2])
plt.plot(DDPG['Step'], DDPG['Value'], label='MADDPG',marker='s',markersize=marker_size,markerfacecolor='none',color = colors[3])
plt.plot(SAL['Step'], SAL['Value'], label='SAL',marker='x',markersize=marker_size,markerfacecolor='none',color = colors[4])


plt.xlabel('Episode',fontsize=xlabel_size)
plt.ylabel('Average episode reward',fontsize=ylabel_size)

# 设置横轴记号
xticks(np.linspace(0,2000,5,endpoint=True),fontsize=xlabel_size)
yticks(None,fontsize=ylabel_size)

xlim(0,2000)

plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.grid()

if os.path.exists('./convergence') is False:
    os.mkdir('./convergence')
plt.rcParams['pdf.fonttype'] = 42
plt.savefig('./convergence/convergence.pdf', bbox_inches='tight')
plt.savefig('./convergence/convergence.eps', bbox_inches='tight')