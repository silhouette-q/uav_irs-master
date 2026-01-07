import os
import sys

# 将项目根目录添加到Python路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import torch.backends.cudnn
from datetime import datetime
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter

# --- MODIFIED: 从修改后的环境文件中导入 ---
# 确保你使用的是简化版的 env.py
from environment.env import UAVEnvFixed, NUM_UAV

from SAC.MASAC import MASAC
from utilities.data_structures.Config import Config


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
    config.seed = 318  # *** MODIFIED: 与环境的随机种子保持一致 ***

    # 创建环境实例
    config.environment = UAVEnvFixed(episode_steps=100)  # *** MODIFIED: 明确指定步数 ***

    config.num_episodes_to_run = 5000  # 初始可以设小一点，比如5000，方便调试
    if num_episodes_to_run is not None:
        config.num_episodes_to_run = num_episodes_to_run

    config.use_GPU = True
    config.randomise_random_seed = False  # *** MODIFIED: 关闭随机化，保证实验可复现 ***
    config.save_model = True  # *** MODIFIED: 建议开启模型保存 ***
    config.agent_num = NUM_UAV

    # 日志和测试频率
    config.log_interval = 100
    config.test_frequency = 500  # 每500个episode测试一次
    config.episode_per_test = 10
    config.deterministic_eval = True  # 在评估时使用确定性策略

    # --- MODIFIED: 修改TensorBoard日志路径以适应您的系统 ---
    # 请根据您的实际情况修改此路径
    # 例如：'G:/CS_Projects/uav_irs_project/uav_irs-master/SAC/runs'
    # 注意使用正斜杠 '/'
    base_log_dir = 'G:/CS_Projects/uav_irs_project/uav_irs-master/SAC/runs'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logging directory: {log_dir}")
    config.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    # --- *** 核心修改点: 更新智能体的观测和动作空间维度 *** ---
    # 获取环境的真实空间定义
    obs_space = config.environment.observation_space
    action_space = config.environment.action_space

    # 打印维度以供检查
    print(f"Environment Observation Space Shape: {obs_space.shape}")
    print(f"Environment Action Space Shape: {action_space.shape}")

    # 使用真实维度来配置智能体
    config.agents_config_dict = [
        {
            'obs_space': obs_space,
            'action_space': action_space
        } for _ in range(config.agent_num)
    ]
    # --- *** 核心修改点结束 *** ---

    config.hyperparameters = {
        "gradient_clipping_norm": 5.0,  # 可以适当调大
        "discount_rate": 0.99,  # 对于长回合任务，通常需要更高的折扣率

        "min_steps_before_learning": 1000,  # *** MODIFIED: 先收集一些数据再开始学习 ***
        "batch_size": 256,
        "update_every_n_steps": 1,  # *** MODIFIED: 每一步都更新，更充分地利用数据 ***
        "learning_updates_per_learning_session": 1,

        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,  # 自动调整时，此项无效

        "Actor": {
            "learning_rate": 3e-4,  # 0.0003
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5.0,
        },

        "Critic": {
            "learning_rate": 3e-4,  # 0.0003
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": int(1e6),  # 100万的经验池
            "tau": 0.005,
            "gradient_clipping_norm": 5.0,
        },
    }

    # 其他配置（保持不变）
    config.file_to_save_data_results = None
    config.file_to_save_results_graph = None
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = False
    config.overwrite_existing_results_file = False

    print("=" * 60)
    print("开始SAC训练 (适配简化版环境)...")
    print(f"- GPU启用: {config.use_GPU}")
    print(f"- 智能体数量: {config.agent_num}")
    print(f"- 训练回合数: {config.num_episodes_to_run}")
    print(f"- 经验池大小: {config.hyperparameters['Critic']['buffer_size']}")
    print(f"- 折扣率: {config.hyperparameters['discount_rate']}")
    print("=" * 60)

    masac = MASAC(config)
    masac.set_random_seeds(config.seed)  # 使用config中的种子
    masac.train()

    print("Training finished.")
    config.writer.close()  # *** MODIFIED: 训练结束后关闭writer ***


if __name__ == '__main__':
    # 运行24000个回合的训练
    train_sac(num_episodes_to_run=24000)