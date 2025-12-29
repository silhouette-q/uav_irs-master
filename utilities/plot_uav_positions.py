import os

import matplotlib.pyplot as plt
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


def plot_uav_positions(positions, fig_name):
    assert len(positions.shape) == 3
    episode_len = positions.shape[0]
    uav_num = positions.shape[1]
    
    if  positions.shape[2] == 28:
        positions = positions[:,0,:-4].reshape(-1,8,3)

    plt.subplot(projection='3d')
    for i in range(uav_num):
        plt.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2])

    coll_pcs = detect_collison(postions=positions)
    for i in range(len(coll_pcs)):
        plt.plot(coll_pcs[i][:,0], coll_pcs[i][:,1], coll_pcs[i][:,2], '*')
    print(coll_pcs)
    plt.savefig(os.path.join('./', fig_name) + '.jpg')
    
    print(os.path.join('./', fig_name) + '.jpg')
    plt.show()
