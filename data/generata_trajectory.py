import os
import sys
sys.path.append(os.getcwd())
sys.path.append('/home/weilai_22/workspace/uav-irs/MAAS')

from Base_Agent import Base_Agent
from nn_builder.pytorch.NN import NN
from utilities.plot_uav_positions import plot_uav_positions
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.Gravity_Exploration_Noise import add_gravity_exploration_noise, cal_ex_reward
from environment.env import UavIrsEnv
from MAAS import MAAS
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import numpy as np
import torch
from datetime import datetime
import argparse


from PPO.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from PPO.config import get_config
from PPO.train.train import parse_args


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_model(path, agent_num):
    model_list = []
    for i in range(agent_num):
        model_list.append(torch.load(path+f'actor_{i}.pth'))
    return model_list


def random_actor(actor_num):
    return torch.rand(actor_num, 4)*2-1


# 生成用于评估的轨迹，轨迹可以被多次利用，减少计算量
def generate_trajectory(gen_list, save_path='/home/weilai_22/workspace/uav-irs/data/trajectory', episode_num=50, episode_len=51, uav_num=8):
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    env = UavIrsEnv()
    env2 = UavIrsEnv()
    env.max_time = episode_len
    env2.max_time = episode_len
    root_path = '/home/weilai_22/workspace/uav-irs/'
    for n in gen_list:
        print(n)
        if n == 'Random':
            pass
        elif n in {'MAAS'}:
            model_list = [torch.load(os.path.join(
                root_path, f'{n}/model/')+f'actor_{i%8}.pth')for i in range(uav_num)]
        elif n in {'DDPG'}:
            model_list = []
            for i in range(uav_num):
                model = NN(input_dim=28, layers_info=[
                           256, 256, 4], output_activation='tanh').cuda()
                model.load_state_dict(torch.load(os.path.join(
                    root_path, f'{n}/model/')+f'actor_{i}.pth'))
                model_list.append(model)
        elif n in {'SAC'}:
            model_list = []
            for i in range(uav_num):
                model = NN(input_dim=28, layers_info=[
                           256, 256, 8], output_activation='none').cuda()
                model.load_state_dict(torch.load(os.path.join(
                    root_path, f'{n}/model/')+f'actor_{i}.pth'))
                model_list.append(model)
        elif n in {'SAL'}:
            model_list = []
            for i in range(uav_num):
                model = NN(input_dim=28, layers_info=[
                           256, 256, 8], output_activation='none').cuda()
                model.load_state_dict(torch.load(os.path.join(
                    root_path, f'{n}/model/')+f'actor_{i}.pth'))
                model_list.append(model)
        elif n in {'PPO'}:
            from gym.spaces import Box
            policy = []
            for agent_id in range(uav_num):
                # policy network
                parser = get_config()
                all_args = parse_args(None, parser)
                po = RMAPPOPolicy(
                    all_args,
                    obs_space=Box(low=-np.inf, high=+np.inf,
                                  shape=(28,), dtype=np.float32,),
                    cent_obs_space=Box(
                        low=-np.inf, high=+np.inf, shape=(28*8,), dtype=np.float32,),
                    act_space=Box(low=-np.inf, high=+np.inf,
                                  shape=(4,), dtype=np.float32,),
                    device='cuda',
                )
                policy.append(po)

            for agent_id in range(uav_num):
                policy_actor_state_dict = torch.load(
                    "./PPO/results/MyEnv/MyEnv/mappo/check/run1/models/actor_agent" + str(agent_id) + ".pt")
                policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

        else:
            print(n)
            raise BaseException

        torch.manual_seed(17)
        np.random.seed(17)
        tra_list = []
        v = []

        eval_rnn_states = np.zeros((1, uav_num, 0, 64), dtype=np.float32,)
        eval_masks = np.ones((1, uav_num, 1), dtype=np.float32)

        for i in range(episode_num):
            state = env.reset()
            state2 = env2.reset()
            current = torch.zeros((uav_num, 1)).cuda()
            current2 = torch.zeros((uav_num, 1)).cuda()
            for j in range(episode_len):
                pos = torch.cat([torch.cat([current, state[0, :uav_num*3].view(uav_num, 3)],
                                dim=1).view(1, -1), state[0, -4:].unsqueeze(0)], dim=1)
                pos2 = torch.cat([torch.cat([current2, state2[0, :uav_num*3].view(
                    uav_num, 3)], dim=1).view(1, -1), state[0, -4:].unsqueeze(0)], dim=1)
                tra_list.append(torch.cat([pos[:, :-4], pos2], dim=1))
                if n == 'Random':
                    action = random_actor(uav_num).numpy()
                    action2 = random_actor(uav_num).numpy()
                elif n in {'MAAS'}:
                    ac_list = [m(state[0, :].view(1, -1)) for m in model_list]
                    ac_list2 = [m(state2[0, :].view(1, -1))
                                for m in model_list]
                    action = torch.cat(ac_list).cpu().detach().numpy()
                    action2 = torch.cat(ac_list2).cpu().detach().numpy()
                elif n in {'DDPG'}:
                    ac_list = [m(state[0, :].view(1, -1)) for m in model_list]
                    ac_list2 = [m(state2[0, :].view(1, -1))
                                for m in model_list]
                    action = torch.cat(ac_list).cpu(
                    ).detach().view(uav_num, 4).numpy()
                    action2 = torch.cat(ac_list2).cpu(
                    ).detach().view(uav_num, 4).numpy()
                elif n in {'SAC', 'SAL'}:
                    ac_list = [torch.tanh(
                        m(state[0, :].view(1, -1))[:, :4]) for m in model_list]
                    ac_list2 = [torch.tanh(
                        m(state2[0, :].view(1, -1))[:, :4]) for m in model_list]
                    action = torch.cat(ac_list).cpu(
                    ).detach().view(uav_num, 4).numpy()
                    action2 = torch.cat(ac_list2).cpu(
                    ).detach().view(uav_num, 4).numpy()
                elif n in {'PPO'}:
                    ac_list = [torch.tanh(m.act(np.array(list(state[0, :].cpu().numpy())), eval_rnn_states[:, 0], eval_masks[:, 0], deterministic=True)[
                                          0][0, 0, 0, :].reshape(1, 4)).detach().cpu().numpy() for m in policy]
                    ac_list2 = [torch.tanh(m.act(np.array(list(state2[0, :].cpu().numpy())), eval_rnn_states[:, 0], eval_masks[:, 0], deterministic=True)[
                                           0][0, 0, 0, :].reshape(1, 4)).detach().cpu().numpy() for m in policy]

                    action = np.array(ac_list).reshape(uav_num, 4)
                    action2 = np.array(ac_list2).reshape(uav_num, 4)
                else:
                    print(n)
                    raise BaseException
                current = torch.tensor(action[:, 0]).view(-1, 1).cuda()
                current2 = torch.tensor(action2[:, 0]).view(-1, 1).cuda()
                state, _, _, _ = env.step(action)
                state2, _, _, _ = env2.step(action2)
            v.append(action[:, 1])
            tras = torch.stack(
                tra_list[-episode_len:]).reshape(episode_len, uav_num*8+4)
            if os.path.exists(save_path+f'/{n}/') is False:
                os.mkdir(save_path+f'/{n}/')
            np.savetxt(save_path+f'/{n}/ep_{i}.txt', tras.cpu().numpy())
        print(np.mean(v))


if __name__ == '__main__':
    torch.manual_seed(8)
    np.random.seed(8)
    generate_trajectory(gen_list=['DDPG'])
    # generate_trajectory(gen_list=['PPO','MAAS'])
    # generate_trajectory(gen_list=['DDPG','SAC','SAL','Random','PPO','MAAS'])
