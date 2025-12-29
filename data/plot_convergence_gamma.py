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
ylabel_size = 20
legend_size = 15

root_path = './smooth'
HMCA99 = pd.read_csv(os.path.join(root_path,'MAAS_copy_logs_2023-12-19-17-37-54-lr_0.0003-gamma_0.99-batch_size_256.csv'))
HMCA95 = pd.read_csv(os.path.join(root_path,'MAAS_logs_2023-12-12-20-02-39.csv'))
HMCA90 = pd.read_csv(os.path.join(root_path,'MAAS_copy_logs_2023-12-19-23-47-27-lr_0.0003-gamma_0.9-batch_size_256.csv'))
HMCA85 = pd.read_csv(os.path.join(root_path,'MAAS_copy_logs_2023-12-20-14-01-26-lr_0.0003-gamma_0.85-batch_size_256.csv'))

plt.figure(figsize=(6,4))
plt.plot(HMCA99['Step'], HMCA99['Value'], label=r'$\gamma$ = 0.99',marker='^',markersize=marker_size,markerfacecolor='none',color = colors[0])
plt.plot(HMCA99['Step'], HMCA95['Value'], label=r'$\gamma$ = 0.95',marker='D',markersize=marker_size,markerfacecolor='none',color = colors[1])
plt.plot(HMCA99['Step'], HMCA90['Value'], label=r'$\gamma$ = 0.90',marker='s',markersize=marker_size,markerfacecolor='none',color = colors[2])
plt.plot(HMCA99['Step'], HMCA85['Value'], label=r'$\gamma$ = 0.85',marker='x',markersize=marker_size,markerfacecolor='none',color = colors[3])

plt.xlabel('Episode',fontsize=xlabel_size)
plt.ylabel('Average episode reward',fontsize=ylabel_size)

# 设置横轴记号
xticks(np.linspace(0,2000,5,endpoint=True),fontsize=ylabel_size)
yticks(None,fontsize=ylabel_size)
xlim(0,2000)

plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.grid()

if os.path.exists('./convergence') is False:
    os.mkdir('./convergence')
    
plt.savefig('./convergence/convergence_gamma.pdf', bbox_inches='tight')
plt.savefig('./convergence/convergence_gamma.eps', bbox_inches='tight')