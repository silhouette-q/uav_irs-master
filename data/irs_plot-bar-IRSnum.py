import os
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from matplotlib import rcParams
from paint_config import *

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    # "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)
plt.get_cmap('Set3')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,6))
marker_size = 10
title_size = 28
xlabel_size = 28
ylabel_size = 29
legend_size = 24

colors = [(55/255, 103/255, 149/255),
          (114/255, 188/255, 213/255),
          (255/255, 208/255, 111/255),
          (231/255, 32/255, 84/255)]

# colors = sns.color_palette("colorblind", n_colors=4)

irs60_res = {'PPO': (281134.0502856999, 266272.37295017985), 'DDPG': (244138.27765596544, 229471.65211418347), 'MAAS': (330394.5525107621, 314175.30051999266)}
irs80_res = {'PPO': (480199.3024684793, 456058.1846564096), 'DDPG': (534213.5945034805, 507951.56741665216), 'MAAS': (562619.2997067282, 535527.837429478)}
irs100_res = {'PPO': (716334.8826380515, 682098.3119934924), 'DDPG': (790173.2046679637, 756204.5362169715), 'MAAS': (828806.5662919452, 792528.7836622295)}

al_set = ['DDPG', 'MAAS', 'PPO']
uav_list = [irs60_res, irs80_res, irs100_res]
obj1 = {al: [res[al][0] for res in uav_list] for al in al_set}
obj2 = {al: [res[al][1] for res in uav_list] for al in al_set}

x = (8)

# 生成一些示例数据
bar_width = 3.5  # 柱的宽度
interval = 0.5
x_values = np.arange(60 - 1.5 * (bar_width + interval), 101, 20)  # 假设有5个x坐标
bar_positions = [x_values, x_values + (bar_width + interval), x_values + 2 * (bar_width + interval), x_values + 3 * (bar_width + interval)]

plt.bar(bar_positions[0], obj1['MAAS'], label='HMCA', color=colors[0], width=bar_width,hatch='///',linewidth=1,edgecolor='black')
plt.bar(bar_positions[1], obj2['MAAS'], label='HMCA-SO', color=colors[1], width=bar_width,hatch='xxx',linewidth=1,edgecolor='black')
plt.bar(bar_positions[2], obj1['PPO'], label='MAPPO', color=colors[2], width=bar_width,hatch='\\\\\\',linewidth=1,edgecolor='black')
plt.bar(bar_positions[3], obj2['PPO'], label='MAPPO-SO', color=colors[3], width=bar_width,hatch='++',linewidth=1,edgecolor='black')

xticks(np.linspace(60, 100, 3, endpoint=True), fontsize=xlabel_size)
yticks(None, fontsize=ylabel_size)

plt.xlabel('The number of reflecting elements', fontsize=xlabel_size)
plt.ylabel('Average secrecy rate [bps]', fontsize=ylabel_size)
plt.legend(fontsize=legend_size, frameon=True, framealpha=1)
plt.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y')
plt.grid()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig('./bench/irs-irsnum.eps', bbox_inches='tight')
plt.savefig('./bench/irs-irsnum.pdf', bbox_inches='tight')
plt.close()
