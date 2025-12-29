import pandas as pd
import numpy as np
import os

def smooth(root_path, csv_file, weight=0.65):
    data = pd.read_csv(filepath_or_buffer=os.path.join(root_path,csv_file),header=0,names=['Step','Value'],dtype={'Step':np.int32,'Value':np.float64})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    save.to_csv('./data/smooth/'+ csv_file)

if __name__=='__main__':
    smooth(root_path='data/raw', csv_file = 'DDPG_logs_2023-12-12-15-40-14.csv')
    smooth(root_path='data/raw', csv_file = 'MAAS_logs_2023-12-12-20-02-39.csv')
    smooth(root_path='data/raw', csv_file = 'PPO_results_MyEnv_MyEnv_mappo_check_run1_logs.csv')
    smooth(root_path='data/raw', csv_file = 'SAC_logs_2023-12-12-15-36-57.csv')
    smooth(root_path='data/raw', csv_file = 'SAL_logs_2023-12-12-22-55-40.csv')
    
    smooth(root_path='data/raw', csv_file = 'MAAS_copy_logs_2023-12-19-13-14-56-lr_0.0005-gamma_0.95-batch_size_256.csv')
    smooth(root_path='data/raw', csv_file = 'MAAS_copy_logs_2023-12-19-15-20-10-lr_0.0001-gamma_0.95-batch_size_256.csv')
    smooth(root_path='data/raw', csv_file = 'MAAS_copy_logs_2023-12-19-17-37-54-lr_0.0003-gamma_0.99-batch_size_256.csv')
    smooth(root_path='data/raw', csv_file = 'MAAS_copy_logs_2023-12-19-23-47-27-lr_0.0003-gamma_0.9-batch_size_256.csv')
    smooth(root_path='data/raw', csv_file = 'MAAS_copy_logs_2023-12-20-14-01-26-lr_0.0003-gamma_0.85-batch_size_256.csv')
