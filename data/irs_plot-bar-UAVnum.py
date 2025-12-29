import os
from matplotlib import pyplot as plt 
import numpy as np
from pylab import *
from matplotlib import rcParams
from paint_config import *
config = {
    "font.family":'Times New Roman',  # 设置字体类型
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

uav4_res = {'PPO': (71049.24135685766, 67131.2435262446), 'DDPG': (83470.64193649903, 78194.667277604),  'MAAS': (88314.94705920009, 83812.95734220768)}
uav5_res = {'PPO': (112621.69592141862, 105957.37721953126), 'DDPG': (128264.07012623653, 121629.37020701829),  'MAAS': (136125.84954477, 129044.64734009995)}
uav6_res = {'PPO': (159104.9163623456, 151592.61797369274), 'DDPG': (181977.04621293084, 171775.66985518814),  'MAAS': (192748.14809140447, 183914.73343851542)}
uav7_res = {'PPO': (218016.42549926194, 207896.38996095434), 'DDPG': (244138.27765596544, 229471.65211418347),  'MAAS': (258231.38413859232, 245578.68152135902)}
uav8_res = {'PPO': (280543.9735215484, 265099.06570695265), 'DDPG': (313868.0214080211, 296681.73044537293), 'MAAS': (329743.3223653088, 315142.5391221304)}

al_set = ['DDPG','MAAS','PPO']
uav_list = [uav4_res, uav6_res, uav8_res]
obj1 = { al: [res[al][0] for res in uav_list] for al in al_set}
obj2 = { al: [res[al][1] for res in uav_list] for al in al_set}

x = (8)

# 生成一些示例数据
bar_width = 0.35  # 柱的宽度
interval = 0.05
x_values = np.arange(4 - 1.5*(bar_width+ interval), 8.01, 2)  # 假设有5个x坐标
bar_positions = [x_values, x_values + (bar_width+interval), x_values + 2 * (bar_width+interval), x_values + 3 * (bar_width+interval)]

plt.bar(bar_positions[0], obj1['MAAS'], label='HMCA', color=colors[0], width=bar_width,hatch='///',linewidth=1,edgecolor='black')
plt.bar(bar_positions[1], obj2['MAAS'], label='HMCA-SO', color=colors[1], width=bar_width,hatch='xxx',linewidth=1,edgecolor='black')
plt.bar(bar_positions[2], obj1['PPO'], label='MAPPO', color=colors[2], width=bar_width,hatch='\\\\\\',linewidth=1,edgecolor='black')
plt.bar(bar_positions[3], obj2['PPO'], label='MAPPO-SO', color=colors[3], width=bar_width,hatch='++',linewidth=1,edgecolor='black')

xticks(np.linspace(4,8,3,endpoint=True),fontsize=xlabel_size)
yticks(None,fontsize=ylabel_size)

plt.xlabel('The number of UAVs',fontsize=xlabel_size)
plt.ylabel('Average secrecy rate [bps]',fontsize=ylabel_size)
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
plt.grid()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig('./bench/irs.eps', bbox_inches='tight')
plt.savefig('./bench/irs.pdf', bbox_inches='tight')
plt.close()
