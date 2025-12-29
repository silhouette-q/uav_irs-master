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
# plt.get_cmap('Set3')

import seaborn as sns
colors = sns.color_palette("colorblind", n_colors=5)

marker_size = 7
title_size = 12
xlabel_size = 12
ylabel_size = 12
legend_size = 12

uav4_res = {'PPO': (79043.083489184, 68882.43992010588), 'DDPG': (83470.64193649903, 78194.667277604),  'MAAS': (88314.94705920009, 86004.56737802459)}
uav5_res = {'PPO': (112621.69592141862, 109259.86611218458), 'DDPG': (128264.07012623653, 121629.37020701829),  'MAAS': (136125.84954477, 132155.85947083723)}
uav6_res = {'PPO': (170220.6075923346, 155167.92050921713), 'DDPG': (181977.04621293084, 171775.66985518814),  'MAAS': (192748.14809140447, 188442.200059193)}
uav7_res = {'PPO': (229345.2279262084, 213108.2933040193), 'DDPG': (244138.27765596544, 229471.65211418347),  'MAAS': (258231.38413859232, 251052.85420724892)}
uav8_res = {'PPO': (290838.6341644618, 272816.31949184864), 'DDPG': (313868.0214080211, 296681.73044537293), 'MAAS': (330588.99676096934, 322401.96759077953)}

al_set = ['DDPG','MAAS']
uav_list = [uav4_res,uav5_res,uav6_res,uav7_res,uav8_res]
obj1 = { al: [res[al][0] for res in uav_list] for al in al_set}
obj2 = { al: [res[al][1] for res in uav_list] for al in al_set}


x = (4,5,6,7,8)
# plt.figure(figsize=(8,6))
plt.plot(x,obj1['MAAS'],label='HMCA',marker='^', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['DDPG'],label='MADDPG',marker='s', markersize = marker_size,markerfacecolor='none')
# plt.plot(x,obj1['PPO'],label='MAPPO',marker='o', markersize = marker_size,markerfacecolor='none')

plt.plot(x,obj2['MAAS'],label='HMCA-SO',marker='D', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['DDPG'],label='MADDPG-SO',marker='x', markersize = marker_size,markerfacecolor='none')
# plt.plot(x,obj2['PPO'],label='MAPPO-SO',marker='*', markersize = marker_size,markerfacecolor='none')

# plt.plot(x,obj1['SAC'],label='MASAC',marker='D', markersize = marker_size,markerfacecolor='none')
# plt.plot(x,obj1['SAL'],label='SAL',marker='x', markersize = marker_size,markerfacecolor='none')
# plt.plot(x,obj1['Random'],label='Random',marker='*', markersize = marker_size,markerfacecolor='none')

plt.xlabel('The number of UAVs',fontsize=xlabel_size)
plt.ylabel('Average secrecy rate [bps]',fontsize=ylabel_size)
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
plt.grid()
xticks(np.linspace(4,8,5,endpoint=True))
xlim(3.9,8.1)
plt.savefig('./data/bench/irs.eps', bbox_inches='tight')
plt.savefig('./data/bench/irs.pdf', bbox_inches='tight')
plt.close()

