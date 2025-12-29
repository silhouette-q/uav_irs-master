import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from datetime import datetime
from gym.spaces import Box

from environment.env import UAVEnvFixed
from SAC.MASAC import MASAC
from utilities.data_structures.Config import Config
from torch.utils.tensorboard import SummaryWriter


def train_sac(num_episodes_to_run=None):
    config = Config()
    config.seed = 1
    config.add_gravity_noise = False
    config.add_bonus = False
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
    config.use_GPU = False
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = False
    # 从环境导入 UAV 数量
    from environment.env import NUM_UAV
    config.agent_num = NUM_UAV
    config.test_frequency = 1e5
    config.episode_per_test = 10
    config.log_interval = 100
    config.deterministic_eval = False

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
        "discount_rate": 0.9,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": False,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "alpha_learning_rate": 0.0003,

        "min_steps_before_learning": 0,
        "batch_size": 256,
        "update_every_n_steps": 12,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": 0.02,
        "add_extra_noise": False,
        "add_gravity_noise": False,
        "do_evaluation_iterations": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 1,
            "initialiser": "orthogonal"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 1,
            "initialiser": "orthogonal"
        },
    }


    masac = MASAC(config)
    masac.set_random_seeds(8)
    masac.load_model()
    # 使用现有的eval方法而不是eval_with_graph
    print("Testing loaded model...")
    # masac.eval_with_graph()  # 这个方法在当前版本中不存在

    # 手动运行一些测试episode
    results = []
    for episode in range(5):
        masac.reset_game()
        episode_reward = 0
        steps = 0
        while not masac.done and steps < 100:
            masac.actions = masac.pick_action(True)  # 确定性策略
            masac.conduct_action(masac.actions, eval=True)
            episode_reward += masac.reward.sum()
            masac.state = masac.next_state
            steps += 1
        results.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    print(f"\nTest Results - Mean Reward: {sum(results)/len(results):.2f}")
    print(f"Best Episode: {max(results):.2f}, Worst Episode: {min(results):.2f}")


if __name__ == '__main__':
    train_sac(4000)
