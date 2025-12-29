import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

from datetime import datetime

from gym.spaces import Box

from DDPG.MADDPG import MADDPG
from environment.env import UavIrsEnv
from utilities.data_structures.Config import Config
from tensorboardX import SummaryWriter


def train_ddpg(num_episodes_to_run=None):
    config = Config()
    config.seed = 4

    config.environment = UavIrsEnv()
    config.num_episodes_to_run = 50
    if num_episodes_to_run is not None:
        config.num_episodes_to_run = num_episodes_to_run

    config.file_to_save_data_results = "../../results/data_and_graphs/Cart_Pole_Results_Data.pkl"
    config.file_to_save_results_graph = "../../results/data_and_graphs/Cart_Pole_Results_Graph.png"
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = False
    config.standard_deviation_results = 1.0
    config.use_GPU = True
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = False
    config.irs_num = config.environment.irs_element_num
    config.agent_num = config.environment.uav_num
    config.test_frequency = 25
    config.episode_per_test = 10
    config.log_interval = 100

    config.writer = SummaryWriter(log_dir= os.path.join(os.path.dirname(os.path.abspath(__file__)) + f"/logs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"), flush_secs=60)
    state = config.environment.reset()
    config.agents_config_dict = [{'obs_space': Box(-1, 1, (state[0, :].shape[0],)), 'action_space': Box(-1, 1, (4,))} for i in
                                 range(config.agent_num)]

    config.hyperparameters = {
        "discount_rate": 0.95,
        "batch_norm": False,
        "normalise_rewards": False,
        "gradient_clipping_norm": 1,
        "action_noise_clipping_range": 0.5,
        "action_noise_std": 0.1,
        "epsilon_decay_rate_denominator": 1,
        "clip_rewards": False,
        "min_steps_before_learning": 0,
        "batch_size": 128,
        "update_every_n_steps": 5,
        "learning_updates_per_learning_session": 1,
        "add_gravity_noise": False,
        
        "do_evaluation_iterations": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": 'TANH',
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

    maddpg = MADDPG(config)
    maddpg.set_random_seeds(18)
    maddpg.train()
    maddpg.save_model()
    maddpg.load_model()
    # maddpg.eval_with_graph()


if __name__ == '__main__':
    train_ddpg(2001)
