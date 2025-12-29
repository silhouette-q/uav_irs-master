from copy import deepcopy

import numpy as np

# some var needed in cal_reward
import torch
from torch import Tensor, from_numpy

device = torch.device('cuda:0')
k_irs = 1
Ntheta = 181
Nphi = 361
theta = np.linspace(0, np.pi, Ntheta)
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)
phi = np.linspace(-np.pi, np.pi, Nphi)
sin_phi = np.sin(phi)
cos_phi = np.cos(phi)
Rx = np.zeros((Ntheta, Nphi))
Ry = np.zeros((Ntheta, Nphi))
Rz = np.zeros((Ntheta, Nphi))
for i in range(Ntheta):
    for j in range(Nphi):
        Rx[i, j] = sin_theta[i] * cos_phi[j]
        Ry[i, j] = sin_theta[i] * sin_phi[j]
        Rz[i, j] = cos_theta[i]

Rx = from_numpy(Rx).cuda()
Ry = from_numpy(Ry).cuda()
Rz = from_numpy(Rz).cuda()
theta = np.linspace(0, np.pi, Ntheta)
sin_theta = np.sin(theta)
sin_theta_map = from_numpy(np.transpose(np.array([sin_theta for _ in range(Nphi)]))).cuda()
delta_T = 0.5


def MultiDiscreteToBox(md, observation_space):
    md = np.array(md, dtype=float) / (observation_space.nvec - 1) * 2 - 1
    return md


def BoxToReal(var, lb, rb):
    lb = np.array(lb)
    rb = np.array(rb)
    var = (var + 1) / 2 * (rb - lb) + lb
    return var


def RealToBox(var, lb, rb):
    lb = np.array(lb)
    rb = np.array(rb)
    var = (var - lb) / (rb - lb) * 2 - 1
    return var


def Calculate_Object(uav_pos_old, uav_pos_no_flatten, irs_beamforming_vector, user_pos, evsdp_pos, sub_opti = False):
    uav_pos = uav_pos_no_flatten.T.flatten()
    f = 2.4E9
    lam = 3E8 / f
    k = 2 * np.pi / lam
    I0 = 1
    B = 2E6
    P_uav = 0.1
    rou = np.power([10], -30 / 10)
    rou = torch.from_numpy(rou).cuda()
    sigma2 = np.power(10, -18.5) * B
    alpha = 2
    alpha_to_user = 2.7
    alpha_dir_to_user = 3.6
    irs_postion = Tensor([1500, 1500, 20]).cuda()
    irs_beamforming_vector = Tensor(irs_beamforming_vector).cuda()
    uav_pos_old = torch.Tensor(uav_pos_old).cuda()
    uav_pos = torch.Tensor(uav_pos).cuda()
    user_pos = torch.Tensor(user_pos).cuda()
    evsdp_pos = torch.Tensor(evsdp_pos).cuda()
    uav_num = uav_pos.shape[0] // 4
    # geneate uav pos and cal uav_center
    uav_I = (uav_pos[:uav_num])
    uav_x = (uav_pos[uav_num:uav_num * 2])
    uav_y = (uav_pos[uav_num * 2:uav_num * 3])
    uav_z = (uav_pos[uav_num * 3:uav_num * 4])

    uav_center = torch.Tensor([torch.mean(uav_x), torch.mean(uav_y), torch.mean(uav_z)]).cuda()
    uav_center_return = uav_center.cpu().numpy()
    uav_x = uav_x - uav_center[0]
    uav_y = uav_y - uav_center[1]
    uav_z = uav_z - uav_center[2]

    # generate pos and consider uav_center as (0,0,0)
    user_pos = user_pos - uav_center
    evsdp_pos = evsdp_pos - uav_center
    irs_pos = irs_postion - uav_center
    uav_center = uav_center - uav_center

    # calculate AF,G
    Ntheta = 181
    Nphi = 361

    irs_angel1_180, irs_angel2_180 = cal_angel(uav_center, irs_pos)
    irs_angel1_pi = irs_angel1_180 / 180 * np.pi
    irs_angel2_pi = irs_angel2_180 / 180 * np.pi
    Ip = I0 * (torch.exp(
        -1j * k * (torch.sin(irs_angel1_pi) * torch.cos(irs_angel2_pi) * uav_x + torch.sin(irs_angel1_pi) * torch.sin(
            irs_angel2_pi) * uav_y + torch.cos(irs_angel1_pi) * uav_z)))
    Inp = Ip * uav_I
    AF_Uniform = torch.zeros((Ntheta, Nphi), device=device)

    for n in range(uav_num):
        AF_Uniform = AF_Uniform + Inp[n] * torch.exp(1j * k * (uav_x[n] * Rx + uav_y[n] * Ry + uav_z[n] * Rz))

    G_denominator = torch.sum(
        torch.sum((torch.abs(torch.pow(AF_Uniform, 2))) * torch.pi / 180 * 2 * torch.pi / 360 * sin_theta_map))

    af_irs = AF_Uniform[irs_angel1_180.int(), irs_angel2_180.int() + 180]

    # cal object 1
    std = Tensor([0.5]).cuda()

    irs_element_num = irs_beamforming_vector.shape[0]
    epsilon_UAV_IRS = 10
    epsilon = 10

    ch1_los = cal_irs_channel(uav_center, irs_pos, irs_element_num)
    ch1 = ch1_los * torch.sqrt(rou) / torch.sqrt(torch.pow(torch.linalg.norm(uav_center - irs_pos), alpha))

    tmp = torch.normal(0, std) + 1j * torch.normal(0, std)
    ch2_los = cal_irs_channel(user_pos, irs_pos, irs_element_num)
    ch2 = np.sqrt(epsilon / (1 + epsilon)) * ch2_los + np.sqrt(1 / (1 + epsilon)) * tmp
    ch2 = ch2 * torch.sqrt(rou) / torch.sqrt(torch.pow(torch.linalg.norm(user_pos - irs_pos), alpha_to_user))

    tmp = torch.normal(0, std) + 1j * torch.normal(0, std)
    ch2e_los = cal_irs_channel(evsdp_pos, irs_pos, irs_element_num)
    ch2e = np.sqrt(epsilon / (1 + epsilon)) * ch2e_los + np.sqrt(1 / (1 + epsilon)) * tmp
    ch2e = ch2e * torch.sqrt(rou) / torch.sqrt(torch.pow(torch.linalg.norm(evsdp_pos - irs_pos), alpha_to_user))

    user_angel1_180, user_angel2_180 = cal_angel(uav_center, user_pos)
    evsdp_angel1_180, evsdp_angel2_180 = cal_angel(uav_center, evsdp_pos)

    tmp = torch.normal(0, std) + 1j * torch.normal(0, std)
    ch3 = tmp * AF_Uniform[user_angel1_180.int(), user_angel2_180.int() + 180] * torch.sqrt(rou) / torch.sqrt(
        torch.pow(torch.linalg.norm(user_pos), alpha_dir_to_user))

    tmp = torch.normal(0, std) + 1j * torch.normal(0, std)
    ch3e = tmp * AF_Uniform[evsdp_angel1_180.int(), evsdp_angel2_180.int() + 180] * torch.sqrt(rou) / torch.sqrt(
        torch.pow(torch.linalg.norm(evsdp_pos), alpha_dir_to_user))

    ch1_conj_T = ch1.view((1, -1)).T.flatten()
    irs_beamforming_vector = (1 / ch1_los.view((1, -1)).T.flatten() * 1 / ch2_los)
    
    if sub_opti is True:
        irs_beamforming_vector = torch.exp(1j * torch.randint(0,8,irs_beamforming_vector.shape)/8 * 2 * np.pi).cuda()
        for i in range(irs_beamforming_vector.shape[0]):
            step = torch.exp(1j * torch.linspace(0,7,8)/8 * 2 * np.pi).cuda()
            tmp_irs_beamforming_vector = irs_beamforming_vector.unsqueeze(0).repeat(8,1)
            tmp_irs_beamforming_vector[:,i] = step
            max_index = torch.argmax(torch.pow(torch.abs(torch.sum(af_irs * ch1_los.view((1, -1)).T.flatten() * tmp_irs_beamforming_vector * ch2_los, axis=1)),2))
            irs_beamforming_vector = tmp_irs_beamforming_vector[max_index]
            
    ch_user = ch3 + torch.sum(af_irs * ch1_conj_T * irs_beamforming_vector * ch2)
    ch_evsdp = ch3e + torch.sum(af_irs * ch1_conj_T * irs_beamforming_vector * ch2e)
    g_user = 4 * np.pi * torch.pow(torch.abs(ch_user), 2) / G_denominator
    g_evsdp = 4 * np.pi * torch.pow(torch.abs(ch_evsdp), 2) / G_denominator

    Pi = torch.pow(uav_I, 2) * P_uav
    R_user = B * torch.log2(1 + sum(Pi) * g_user / sigma2)
    R_evsdp = B * torch.log2(1 + sum(Pi) * g_evsdp / sigma2)
    obj_1 = R_user - R_evsdp

    # cal object 2
    a = torch.max(torch.abs(AF_Uniform), dim=1)[0]
    sll = 0
    ml = 0

    for i, l in enumerate(a):
        if i != 0 and i != a.shape[0] - 1 and a[i] > a[i - 1] and a[i] > a[i + 1]:
            if l > ml:
                sll = ml
                ml = l
            elif l > sll:
                sll = l
    obj_2 = 20 * torch.log10(torch.abs(sll / ml))

    # cal object 3
    obj_3 = cal_obj3(uav_pos_old, uav_pos_no_flatten)
    return obj_1.cpu(), obj_2.cpu(), obj_3.cpu(), uav_center_return


def cal_obj3(uav_origin, uav):
    uav = torch.Tensor(uav).cuda()
    dis = torch.linalg.norm(uav_origin[:, 1:] - uav[:, 1:], axis=1)
    form_time = delta_T
    V_ad = dis / form_time
    V_ver = (uav[:, -1:] - uav_origin[:, -1:]) / form_time

    Utip = 120
    v0 = 4.03
    d0 = 0.6
    s = 0.05
    rho = 1.225
    A = 0.503
    delta = 0.012
    omega = 300
    R = 0.4
    W = 20
    k = 0.1
    g = 9.8

    Po = delta / 8 * rho * s * A * np.power(omega, 3) * np.power(R, 3)
    Pi = (1 + k) * np.sqrt(np.power(W, 3)) / np.sqrt(2 * rho * A)
    Pv = Po * (1 + 3 * torch.pow((V_ad / Utip), 2)) + Pi * (
        torch.sqrt(
            torch.sqrt(1 + (torch.pow(V_ad, 4) / 4 / np.power(v0, 4))) - torch.pow(V_ad, 2) / 2 / np.power(v0, 2))) \
         + 0.5 * d0 * rho * s * A * torch.pow(V_ad, 3)

    E = (Pv.view(-1, 1) + W * V_ver) * form_time

    return E


def cal_angel(pos1, pos2):
    pos2 = deepcopy(pos2 - pos1)

    molecule = torch.linalg.norm(pos2[0:2])
    denominator = torch.abs(pos2[2])
    if pos2[2] < 0:
        user_an_1 = torch.pi - torch.arctan(molecule / denominator)
    else:
        user_an_1 = torch.arctan(molecule / denominator)

    side1 = pos2[1]
    side2 = pos2[0]

    if side1 > 0 and side2 > 0:
        user_an_2 = torch.arctan(abs(side1) / abs(side2))
    elif side1 < 0 and side2 > 0:
        user_an_2 = torch.pi - torch.arctan(abs(side1) / abs(side2))

    elif side1 < 0 and side2 < 0:
        user_an_2 = -torch.pi + torch.arctan(abs(side1) / abs(side2))
    else:
        user_an_2 = -torch.arctan(abs(side1) / abs(side2))

    user_an_1 = torch.round(user_an_1 / np.pi * 180)
    user_an_2 = torch.round(user_an_2 / np.pi * 180)
    return user_an_1, user_an_2


def cal_irs_channel(tar, irs, irs_element_num):
    # cal channel to user
    tmp = torch.arange(0, irs_element_num).cuda()
    dis = torch.linalg.norm(tar - irs)
    spct = (tar[1] - irs[1]) / dis
    a = torch.exp(1j * torch.pi * k_irs * (tmp * spct))
    return a
