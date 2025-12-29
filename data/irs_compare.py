import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from environment.util import Calculate_Object
import os
import torch

if __name__ == '__main__':
    bench_uav_num = 4
    for bench_uav_num in range(4,9,1):
        print(bench_uav_num)
        d = {}
        for r, dirs, files in os.walk('data/trajectory'):
            if r == 'data/trajectory':
                continue
            episode_len = 50
            al_n = os.path.split(r)[-1]
            if al_n not in ('DDPG'):
                continue
            print(al_n)
            l1 = []
            random_l1 = []
            collision_times = 0
            for f in files:
                tra = np.loadtxt(os.path.join(r, f))
                user_evsdp = tra[:, -4:].reshape(tra.shape[0], 4)
                uavs_pos = tra[:, :-4].reshape(tra.shape[0], -1, 4)
                user_evsdp = user_evsdp * \
                    np.array([15, 15, 15, 15]) + \
                    np.array([1530, 1450, 1550, 1450])
                uavs_pos = uavs_pos * \
                    np.array([0.5, 50, 50, 15]) + np.array([0.5, 50, 50, 75])

                for j in range(1, episode_len):
                    obj_1, obj_2, obj_3, _ = Calculate_Object(uavs_pos[j-1,:bench_uav_num], uavs_pos[j,:bench_uav_num], torch.ones(60),
                                                            np.concatenate([user_evsdp[j, :2], torch.zeros(1)]), np.concatenate([user_evsdp[j, -2:], torch.zeros(1)]))
                    
                    random_obj1, obj_2, obj_3, _ = Calculate_Object(uavs_pos[j-1,:bench_uav_num], uavs_pos[j,:bench_uav_num], torch.ones(60),
                                                            np.concatenate([user_evsdp[j, :2], torch.zeros(1)]), np.concatenate([user_evsdp[j, -2:], torch.zeros(1)]),sub_opti= True)
                    l1.append(obj_1.numpy())
                    
                    random_l1.append(random_obj1)
                    
            d[al_n] = (np.mean(l1),np.mean(random_l1))
        print(d)
    bench_uav_num = 8
    for irs_num in range(60,101,20):
        print(bench_uav_num)
        d = {}
        for r, dirs, files in os.walk('data/trajectory'):
            if r == 'data/trajectory':
                continue
            episode_len = 50
            al_n = os.path.split(r)[-1]
            if al_n not in ('DDPG'):
                continue
            print(al_n)
            l1 = []
            random_l1 = []
            collision_times = 0
            for f in files:
                tra = np.loadtxt(os.path.join(r, f))
                user_evsdp = tra[:, -4:].reshape(tra.shape[0], 4)
                uavs_pos = tra[:, :-4].reshape(tra.shape[0], -1, 4)
                user_evsdp = user_evsdp * \
                    np.array([15, 15, 15, 15]) + \
                    np.array([1530, 1450, 1550, 1450])
                uavs_pos = uavs_pos * \
                    np.array([0.5, 50, 50, 15]) + np.array([0.5, 50, 50, 75])

                for j in range(1, episode_len):
                    obj_1, obj_2, obj_3, _ = Calculate_Object(uavs_pos[j-1,:bench_uav_num], uavs_pos[j,:bench_uav_num], torch.ones(irs_num),
                                                            np.concatenate([user_evsdp[j, :2], torch.zeros(1)]), np.concatenate([user_evsdp[j, -2:], torch.zeros(1)]))
                    
                    random_obj1, obj_2, obj_3, _ = Calculate_Object(uavs_pos[j-1,:bench_uav_num], uavs_pos[j,:bench_uav_num], torch.ones(irs_num),
                                                            np.concatenate([user_evsdp[j, :2], torch.zeros(1)]), np.concatenate([user_evsdp[j, -2:], torch.zeros(1)]),sub_opti= True)
                    l1.append(obj_1.numpy())
                    
                    random_l1.append(random_obj1)
                    
            d[al_n] = (np.mean(l1),np.mean(random_l1))
        print(d)
        
        