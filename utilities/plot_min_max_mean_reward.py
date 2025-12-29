import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入数据集
#MAASAC
MAASAC_max = pd.read_csv('~/workspace/uav-irs/data/MAASAC_max.csv')
MAASAC_min = pd.read_csv('~/workspace/uav-irs/data/MAASAC_min.csv')
MAASAC = pd.read_csv('~/workspace/uav-irs/data/MAASAC.csv')

MAASAC['minValue'] = MAASAC_min['Value']
MAASAC['maxValue'] = MAASAC_max['Value']

plt.plot(MAASAC['Step'], MAASAC['Value'], color='deeppink', label='MAASAC')
plt.fill_between(MAASAC['Step'], MAASAC['minValue'], MAASAC['maxValue'], color='deeppink', alpha=0.2)

#MASAC
MASAC_max = pd.read_csv('~/workspace/uav-irs/data/MASAC_max.csv')
MASAC_min = pd.read_csv('~/workspace/uav-irs/data/MASAC_min.csv')
MASAC = pd.read_csv('~/workspace/uav-irs/data/MASAC.csv')

MASAC['minValue'] = MASAC_min['Value']
MASAC['maxValue'] = MASAC_max['Value']

plt.plot(MASAC['Step'], MASAC['Value'], color='cornflowerblue', label='MASAC')
plt.fill_between(MASAC['Step'], MASAC['minValue'], MASAC['maxValue'], color='cornflowerblue', alpha=0.2)

MADDPG_max = pd.read_csv('~/workspace/uav-irs/data/MADDPG_max.csv')
MADDPG_min = pd.read_csv('~/workspace/uav-irs/data/MADDPG_min.csv')
MADDPG = pd.read_csv('~/workspace/uav-irs/data/MADDPG.csv')

MADDPG['minValue'] = MADDPG_min['Value']
MADDPG['maxValue'] = MADDPG_max['Value']

plt.plot(MADDPG['Step'], MADDPG['Value'], color='darkcyan', label='MADDPG')
plt.fill_between(MADDPG['Step'], MADDPG['minValue'], MADDPG['maxValue'], color='darkcyan', alpha=0.2)
plt.legend(loc=4)

plt.xlabel('Episode')
plt.ylabel('Average episode reward')
plt.show()

plt.savefig('./tmp.png')