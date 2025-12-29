import os
from matplotlib import pyplot as plt 
import numpy as np


def detect_collison(postions, uav_num=8):
    real_pos = postions.reshape(-1, uav_num, 3).copy()
    real_pos[:, :, :2] = real_pos[:, :, :2]*50+50
    real_pos[:, :, 2] = real_pos[:, :, 2]*15+75
    coll_pcs = []
    for i in range(uav_num):
        for j in range(i+1, uav_num):
            coll_flag = np.linalg.norm(real_pos[:, i, :]-real_pos[:, j, :], axis=1) < 0.5
            if coll_flag.any():
                coll_pcs.append(postions[coll_flag, j, :])
                coll_pcs.append(postions[coll_flag, i, :])
    return coll_pcs


def plot_uav_positions(positions, save_path):
    assert len(positions.shape) == 3
    episode_len = positions.shape[0]
    uav_num = positions.shape[1]
    assert positions.shape[2] == 3
    plt.get_cmap('Set3')
    plt.subplot(projection='3d')
    for i in range(uav_num):
        plt.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2])

    coll_pcs = detect_collison(postions=positions)
    for i in range(len(coll_pcs)):
        plt.scatter(coll_pcs[i][:,0], coll_pcs[i][:,1], coll_pcs[i][:,2], s=10,marker='*')
    
    if len(coll_pcs)>0:
        print(coll_pcs)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    
    if os.path.exists('trajectory_pic') is False:
        os.makedirs('trajectory_pic')
    uav_num = 8    
    for r,dirs,files in os.walk('trajectory_for_pic'):
        al_n = os.path.split(r)[-1]

        for f in files:
            tra = np.loadtxt(os.path.join(r,f))
            user_evsdp = tra[:,-4:]
            uavs_pos = tra[:,:-4].reshape(tra.shape[0],-1,4)[:,:uav_num,1:]
            uavs_pos[:,:,:2] = uavs_pos[:,:,:2] * 50 + 50
            uavs_pos[:,:,2] = uavs_pos[:,:,2] * 15 + 75
            if os.path.exists(f'trajectory_pic/{al_n}') is False:
                os.makedirs(f'trajectory_pic/{al_n}')
            print(al_n,f)
            plot_uav_positions(uavs_pos,os.path.join(f'trajectory_pic/{al_n}',f+'.png'))