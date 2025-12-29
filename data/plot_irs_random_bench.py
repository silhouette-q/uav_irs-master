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

uav4_res = {'PPO': (79043.85538993032, -0.005574363152403734, 13793.856811523438, 0.9011876136772109), 'DDPG': (85664.29389227786, -0.006072370277581798, 15873.948669433594, 0.999592005212088), 
            'SAL': (68851.7364329573, -0.0054163853898338025, 13037.820434570312, 0.8569200972556101), 'SAC': (76260.82239991032, -0.006006374538138452, 13295.860290527344, 0.9277088180873085), 
            'MAAS': (88351.49772627329, -0.005881980204904591, 12189.020538330078, 0.9816807160665262), 'Random': (24399.23062346849, -0.004459918543090398, 14399.809265136719, 0.5045604602840482)}
uav5_res = {'PPO': (121827.800167067, -0.03471070665977673, 17401.40838623047, 0.9078673971315551), 'DDPG': (130856.63999945569, -0.03562272967865612, 19933.775329589844, 0.9996731954618377), 
            'SAL': (106658.29833267361, -0.03445604261169489, 16315.017700195312, 0.8575633628737913), 'SAC': (118725.27492732184, -0.03657131819786716, 16562.301635742188, 0.9347378075548367), 
            'MAAS': (136444.22702506362, -0.036723763050786576, 15098.992919921875, 0.9806003733500838), 'Random': (37188.81228862857, -0.027320749043845856, 17998.768615722656, 0.5044310123239244)}
uav6_res = {'PPO': (170220.32071621256, -0.10472001227140514, 20809.230041503906, 0.9086058645533339), 'DDPG': (185281.0273380132, -0.1093455797867829, 23411.561584472656, 0.9990407050993978), 
            'SAL': (152488.86143742283, -0.10200302107528882, 19595.11260986328, 0.8632319079191609), 'SAC': (164775.7551441712, -0.10965749432231695, 19697.064208984375, 0.924950136898282), 
            'MAAS': (193699.30627699074, -0.11187287815365617, 18080.528259277344, 0.9812448084469708), 'Random': (52633.4401353005, -0.07982861809406985, 21595.310974121094, 0.5040759036127402)}
uav7_res = {'PPO': (229345.871947666, -0.21938544828669396, 24295.993041992188, 0.9175602474185752), 'DDPG': (252004.91879784357, -0.22963812256500177, 26648.66943359375, 0.9991766131167509), 
            'SAL': (206278.4553901026, -0.21635281328914274, 22814.247131347656, 0.8699614758732893), 'SAC': (223076.7126931719, -0.2264264377686147, 23027.16064453125, 0.9296654932994163), 
            'MAAS': (258692.00519205793, -0.2332605467981674, 21075.559997558594, 0.9798797459338326), 'Random': (71102.6569767745, -0.1692896701847812, 25199.954223632812, 0.5032982413483779)}
uav8_res = {'PPO': (290838.03228698217, -0.36925677840985366, 27795.388793945312, 0.9154007098388238), 'DDPG': (325672.38994943624, -0.393530972676377, 30648.562622070312, 0.999091865693276), 
            'SAL': (263855.6206075978, -0.36945445449570147, 26053.274536132812, 0.8668486300786827), 'SAC': (287931.0431247385, -0.3899392476552621, 26261.248779296875, 0.9285751647648237), 
            'MAAS': (332087.97977719735, -0.39738223889326557, 24093.93310546875, 0.9790791068323745), 'Random': (90779.45532626173, -0.2869695951876954, 28794.943237304688, 0.5022317798283635)}

al_set = ['PPO','DDPG','SAL','SAC','MAAS','Random']
uav_list = [uav4_res,uav5_res,uav6_res,uav7_res,uav8_res]
obj1 = { al: [res[al][0] for res in uav_list] for al in al_set}
obj2 = { al: [res[al][1] for res in uav_list] for al in al_set}
obj3 = { al: [res[al][2] for res in uav_list] for al in al_set}



x = (4,5,6,7,8)
# plt.figure(figsize=(8,6))
plt.plot(x,obj1['MAAS'],label='HMCA',marker='^', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['SAC'],label='MASAC',marker='D', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['DDPG'],label='MADDPG',marker='s', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['PPO'],label='MAPPO',marker='o', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['SAL'],label='SAL',marker='x', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj1['Random'],label='Random',marker='*', markersize = marker_size,markerfacecolor='none')
plt.xlabel('The number of UAVs',fontsize=xlabel_size)
plt.ylabel('Average security rate [bps]',fontsize=ylabel_size)
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
# plt.title('(a)', y=-0.2,fontsize=title_size)
plt.grid()
xticks(np.linspace(4,8,5,endpoint=True))
xlim(3.9,8.1)
plt.savefig('./data/bench/obj1.eps', bbox_inches='tight')
plt.savefig('./data/bench/obj1.jpg', bbox_inches='tight')
plt.close()

plt.plot(x,obj2['MAAS'],label='HMCA',marker='^', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['SAC'],label='MASAC',marker='D', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['DDPG'],label='MADDPG',marker='s', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['PPO'],label='MAPPO',marker='o', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['SAL'],label='SAL',marker='x', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj2['Random'],label='Random',marker='*', markersize = marker_size,markerfacecolor='none')
plt.xlabel('The number of UAVs',fontsize=xlabel_size)
plt.ylabel('Average maximum SLLs [dB]',fontsize=ylabel_size)
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
plt.grid()
# plt.title('(b)', y=-0.2,fontsize=title_size)
xticks(np.linspace(4,8,5,endpoint=True))
xlim(3.9,8.1)
plt.savefig('./data/bench/obj2.eps', bbox_inches='tight')
plt.savefig('./data/bench/obj2.jpg', bbox_inches='tight')
plt.close()



plt.plot(x,obj3['MAAS'],label='HMCA',marker='^', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj3['SAC'],label='MASAC',marker='D', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj3['DDPG'],label='MADDPG',marker='s', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj3['PPO'],label='MAPPO',marker='o', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj3['SAL'],label='SAL',marker='x', markersize = marker_size,markerfacecolor='none')
plt.plot(x,obj3['Random'],label='Random',marker='*', markersize = marker_size,markerfacecolor='none')
plt.xlabel('The number of UAVs',fontsize=xlabel_size)
plt.ylabel('Total energy consumption [J]',fontsize=ylabel_size)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend(fontsize=legend_size,frameon=True,framealpha=1)
# plt.title('(c)', y=-0.2,fontsize=title_size)
xticks(np.linspace(4,8,5,endpoint=True))
xlim(3.9,8.1)
plt.grid()
plt.savefig('./data/bench/obj3.eps', bbox_inches='tight')
plt.savefig('./data/bench/obj3.jpg', bbox_inches='tight')


