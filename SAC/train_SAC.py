import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import torch.backends.cudnn

# GPU优化配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用所有可用的GPU
# 启用GPU加速和优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速矩阵乘法
    torch.backends.cudnn.allow_tf32 = True  # 启用cuDNN的TF32
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("警告: CUDA不可用，将使用CPU训练") 

from datetime import datetime

from gym.spaces import Box

from environment.env import UAVEnvFixed
from environment.env import NUM_UAV  # 额外导入智能体数量
from SAC.MASAC import MASAC
from utilities.data_structures.Config import Config
from torch.utils.tensorboard import SummaryWriter



def train_sac(num_episodes_to_run=None):
    config = Config()
    config.seed = 0

    ######
    config.add_gravity_noise = False
    config.add_bonus = False
    #####
    config.environment = UAVEnvFixed()
    config.num_episodes_to_run = 2000
    if num_episodes_to_run is not None:
        config.num_episodes_to_run = num_episodes_to_run
    config.file_to_save_data_results = "../../results/data_and_graphs/Cart_Pole_Results_Data.pkl"
    config.file_to_save_results_graph = "../../results/data_and_graphs/Cart_Pole_Results_Graph.png"
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = False
    config.standard_deviation_results = 1.0
    config.use_GPU = True  # 启用GPU训练
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = False
    #config.irs_num = config.environment.irs_element_num
    config.agent_num = NUM_UAV
    config.test_frequency = 25
    config.episode_per_test = 10
    config.log_interval = 100

    config.deterministic_eval = True

    # Set up TensorBoard logging with timestamped directories
    base_log_dir = '/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/runs'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print(f"TensorBoard logging directory: {log_dir}")
    config.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    state = config.environment.reset()
    config.agents_config_dict = [{'obs_space': Box(-1, 1, (state[0, :].shape[0],)), 'action_space': Box(-1, 1, (4,))} for i in
                                 range(config.agent_num)]

    config.hyperparameters = {
        "gradient_clipping_norm": 1,
        "discount_rate": 0.95,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": False,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "alpha_learning_rate": 0.0005,

        "min_steps_before_learning": 0,
        "batch_size": 256,
        "update_every_n_steps": 5,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": 0.01,
        "add_extra_noise": False,
        "do_evaluation_iterations": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 20,
            # "initialiser": "xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 20,
            # "initialiser": "xavier"
        },
    }

    # 添加性能监控配置
    config.training_start_time = datetime.now()
    config.episode_times = []  # 存储每个episode的训练时间
    config.performance_monitoring = True
    config.log_memory_usage = True  # 是否记录内存使用情况

    print("="*60)
    print("开始SAC训练...")
    print(f"训练配置:")
    print(f"- GPU启用: {config.use_GPU}")
    print(f"- 智能体数量: {config.agent_num}")
    print(f"- 训练回合数: {config.num_episodes_to_run}")
    print(f"- 批次大小: {config.hyperparameters['batch_size']}")
    print(f"- 学习率: {config.hyperparameters['Actor']['learning_rate']}")
    print(f"- 日志间隔: {config.log_interval}")
    print("="*60)

    masac = MASAC(config)
    masac.set_random_seeds(16)
    masac.train()
    print("Training finished.")
    # masac.eval_with_graph()


if __name__ == '__main__':
    train_sac(24000)
