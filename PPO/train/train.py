# """
# # @Time    : 2021/6/30 10:07 下午
# # @Author  : hezhiqiang
# # @Email   : tinyzqh@163.com
# # @File    : train.py
# """
#
# # !/usr/bin/env python
# import sys
# import os
# import socket
# import setproctitle
# import numpy as np
# from pathlib import Path
# import torch
#
# # Get the parent directory of the current file
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# # Append the parent directory to sys.path, otherwise the following import will fail
# sys.path.append(parent_dir)
#
# from PPO.config import get_config
# #from PPO.envs.env_wrappers import DummyVecEnv
# from environment.env import UAVEnvFixed # <-- 导入你的环境
#
# """Train script for MPEs."""
#
#
#
# # def make_train_env(all_args):
# #     def get_env_fn(rank):
# #         def init_env():
# #             # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
# #             # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
# #
# #             from environment.env import UAVEnvFixed # <-- 导入你的环境
# #
# #             env = UAVEnvFixed()
# #
# #             # from envs.env_discrete import DiscreteActionEnv
# #
# #             # env = DiscreteActionEnv()
# #
# #             env.seed(all_args.seed + rank * 1000)
# #             return env
# #
# #         return init_env
# #
# #     return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
# #
# #
# # def make_eval_env(all_args):
# #     def get_env_fn(rank):
# #         def init_env():
# #             # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
# #             # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
# #             from environment.env import UAVEnvFixed # <-- 导入你的环境
# #
# #             env = UAVEnvFixed()
# #             # from envs.env_discrete import DiscreteActionEnv
# #             # env = DiscreteActionEnv()
# #             env.seed(all_args.seed + rank * 1000)
# #             return env
# #
# #         return init_env
# #
# #     return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
#
#
# def parse_args(args, parser):
#     parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
#     #parser.add_argument("--num_landmarks", type=int, default=3)
#     parser.add_argument("--num_agents", type=int, default=8, help="number of players")
#
#     all_args = parser.parse_known_args(args)[0]
#
#     return all_args
#
#
# def main(args):
#     parser = get_config()
#     all_args = parse_args(args, parser)
#
#     if all_args.algorithm_name == "rmappo":
#         assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
#     elif all_args.algorithm_name == "mappo":
#         assert (
#             all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
#         ), "check recurrent policy!"
#     else:
#         raise NotImplementedError
#
#     assert (
#         all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
#     ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."
#
#     # cuda
#     if all_args.cuda and torch.cuda.is_available():
#         print("choose to use gpu...")
#         device = torch.device("cuda:0")
#         torch.set_num_threads(all_args.n_training_threads)
#         if all_args.cuda_deterministic:
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#     else:
#         print("choose to use cpu...")
#         device = torch.device("cpu")
#         torch.set_num_threads(all_args.n_training_threads)
#
#     # run dir
#     run_dir = (
#         Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
#         / all_args.env_name
#         / all_args.scenario_name
#         / all_args.algorithm_name
#         / all_args.experiment_name
#     )
#     if not run_dir.exists():
#         os.makedirs(str(run_dir))
#
#     if not run_dir.exists():
#         curr_run = "run1"
#     else:
#         exst_run_nums = [
#             int(str(folder.name).split("run")[1])
#             for folder in run_dir.iterdir()
#             if str(folder.name).startswith("run")
#         ]
#         if len(exst_run_nums) == 0:
#             curr_run = "run1"
#         else:
#             curr_run = "run%i" % (max(exst_run_nums) + 1)
#     run_dir = run_dir / curr_run
#     if not run_dir.exists():
#         os.makedirs(str(run_dir))
#
#     setproctitle.setproctitle(
#         str(all_args.algorithm_name)
#         + "-"
#         + str(all_args.env_name)
#         + "-"
#         + str(all_args.experiment_name)
#         + "@"
#         + str(all_args.user_name)
#     )
#
#     # seed
#     torch.manual_seed(all_args.seed)
#     torch.cuda.manual_seed_all(all_args.seed)
#     np.random.seed(all_args.seed)
#
#     # # env init
#     # envs = make_train_env(all_args)
#     # eval_envs = make_eval_env(all_args) if all_args.use_eval else None
#     # num_agents = all_args.num_agents
#     # --- 环境初始化 ---
#     # 直接实例化你的环境
#     envs = UAVEnvFixed()
#     num_agents = envs.agent_num
#     all_args.num_agents = num_agents  # 更新配置中的智能体数量
#     all_args.episode_length = envs.episode_steps  # 更新配置中的回合长度
#     # 移除评估环境相关代码
#     eval_envs = None
#
#     config = {
#         "all_args": all_args,
#         "envs": envs,
#         "eval_envs": eval_envs,
#         "num_agents": num_agents,
#         "device": device,
#         "run_dir": run_dir,
#     }
#
#     # run experiments
#     if all_args.share_policy:
#         from PPO.runner.shared.env_runner import EnvRunner as Runner
#     else:
#         from PPO.runner.separated.env_runner import EnvRunner as Runner
#
#     runner = Runner(config)
#     runner.run()
#
#     # post process
#     envs.close()
#     if all_args.use_eval and eval_envs is not envs:
#         eval_envs.close()
#
#     runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
#     runner.writter.close()
#
#
# if __name__ == "__main__":
#     main(sys.argv[1:])
import os
import sys
from pathlib import Path
import numpy as np
import torch
#import setproctitle

# --- 新增：添加环境目录和项目根目录到 Python 路径 ---
# 获取当前文件的绝对路径 -> train.py
# 获取 train.py 的父目录 -> PPO/train
# 获取 PPO/train 的父目录 -> PPO
# 获取 PPO 的父目录 -> uav_irs-master (项目根目录)
# 获取环境目录 -> uav_irs-master/environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(PPO_DIR)
ENV_DIR = os.path.join(BASE_DIR, 'environment')
sys.path.insert(0, ENV_DIR)
sys.path.insert(0, BASE_DIR) # 确保可以导入 Base_Agent 等
# --- 路径设置结束 ---

from PPO.config import get_config
# from PPO.envs.env_wrappers import SubprocVecEnv, DummyVecEnv # 不再使用并行环境包装器
from PPO.runner.separated.base_runner import Runner # 假设我们修改这个 Runner
from environment.env import UAVEnvFixed # <-- 导入你的环境
from torch.utils.tensorboard import SummaryWriter # <-- 使用标准的 SummaryWriter


# 不再需要 make_train_env 和 make_eval_env 函数

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]

    # if all_args.algorithm_name == "rmappo":
    #     print("不支持 rmappo, 请使用 mappo")
    #     # assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    # elif all_args.algorithm_name == "mappo":
    #     print("你选择了 mappo")
    #     # assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    # else:
    #     raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("选择使用 gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("选择使用 cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir (结果保存目录)
    # 路径结构: PPO/results/环境名/场景名(与环境名相同)/算法名/实验名
    run_dir = Path(PPO_DIR) / "results" / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # --- 设置 TensorBoard writer 路径 ---
    # 选择你想要的运行 ID，例如 'ppo_run1'
    # 日志会保存在 PPO 目录下的 runs/ppo_run1
    log_dir = os.path.join(PPO_DIR, f'runs/1')
    print(f"Tensorboard 日志目录: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    # --- Writer 设置完成 ---

    # 设置进程标题，方便在 top/htop 中查看
    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #                           str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
    #     all_args.user_name))

    # 设置随机种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # --- 环境初始化 ---
    # 直接实例化你的环境
    print("初始化环境 UAVEnvFixed...")
    envs = UAVEnvFixed()
    num_agents = envs.agent_num
    all_args.num_agents = num_agents # 更新配置中的智能体数量
    all_args.episode_length = envs.episode_steps # 更新配置中的回合长度
    print(f"环境初始化完成: 智能体数量={num_agents}, 回合长度={all_args.episode_length}")

    # 获取并设置观察和动作空间到 all_args (Runner 中会用到)
    # 假设所有 agent 的空间相同
    all_args.obs_space = envs.observation_space
    all_args.act_space = envs.action_space
    # PPO的Critic通常也只使用局部观察，这里将 share_obs_space 设为 obs_space
    # 如果你的 PPO 实现需要全局状态给 Critic，你需要修改这里以及环境以提供全局状态
    all_args.share_obs_space = envs.observation_space
    print(f"观察空间: {all_args.obs_space}")
    print(f"动作空间: {all_args.act_space}")
    print(f"共享观察空间 (Critic): {all_args.share_obs_space}")

    # 移除评估环境相关代码
    eval_envs = None
    # --- 环境初始化完成 ---


    config = {
        "all_args": all_args,
        "envs": envs, # <--- 传递单个环境实例
        "eval_envs": eval_envs, # 设置为 None
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir, # 这个是 PPO 内部的结果保存路径
        "writer": writer # <--- 传递 writer 实例
    }

    # 运行实验
    # 注意: Runner 类本身需要被修改以适应单环境和回合制循环
    print("初始化 Runner...")
    runner = Runner(config)
    print("开始训练...")
    runner.run() # 这里的 run 方法需要被重写

    # 训练结束后的处理
    print("训练完成.")
    envs.close()
    # if all_args.use_eval and eval_envs is not envs: # 这部分不再需要
    #     eval_envs.close()

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json")) # 保存 TensorBoard 数据
    writer.close()


if __name__ == "__main__":
    # 注意: 确保你的 PPO/config.py 中定义了必要的参数
    # 你可能需要在这里覆盖一些默认参数，例如:
    # sys.argv.extend(['--env_name', 'UAVEnvFixed', '--scenario_name', 'UAVScenario', '--algorithm_name', 'mappo', '--experiment_name', 'run1', '--seed', '42', '--num_env_steps', '1000000'])
    # '--use_recurrent_policy', 'True' (如果你的模型是RNN)
    main(sys.argv[1:])