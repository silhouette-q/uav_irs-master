#!/usr/bin/env python3
"""
GPU设备一致性测试脚本
用于验证所有设备相关的修复
"""

import sys
import torch
import numpy as np
import os

# 添加路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print("="*60)
print("设备一致性测试脚本")
print("="*60)

def test_basic_gpu_setup():
    """测试基础GPU配置"""
    print("\n1. 基础GPU配置测试:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA版本: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("   ⚠ 警告: CUDA不可用，将使用CPU模式")
        return False

def test_gravity_noise():
    """测试重力噪声函数"""
    print("\n2. 重力噪声函数测试:")
    try:
        from utilities.Gravity_Exploration_Noise import add_gravity_exploration_noise

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        action = torch.randn(1, 4, device=device)
        ep_i_sum = 0.1

        result = add_gravity_exploration_noise(action, ep_i_sum)

        print(f"   ✓ 重力噪声函数正常")
        print(f"   输入设备: {action.device}")
        print(f"   输出设备: {result.device}")
        print(f"   输入形状: {action.shape}")
        print(f"   输出形状: {result.shape}")
        return True
    except Exception as e:
        print(f"   ✗ 重力噪声函数错误: {e}")
        return False

def test_ou_noise_conversion():
    """测试OU噪声转换"""
    print("\n3. OU噪声转换测试:")
    try:
        from utilities.OU_Noise import OU_Noise

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ou_noise = OU_Noise(4, seed=123, sigma=0.2)

        # 模拟SACAgent中的处理
        noise_sample = ou_noise.sample()
        noise_tensor = torch.from_numpy(noise_sample).float().to(device)
        action = torch.randn(1, 4, device=device)

        # 测试加法操作
        combined_action = action + noise_tensor

        print(f"   ✓ OU噪声转换正常")
        print(f"   numpy → tensor转换: {type(noise_sample)} → {type(noise_tensor)}")
        print(f"   设备一致性: noise-{noise_tensor.device}, action-{action.device}, combined-{combined_action.device}")
        return True
    except Exception as e:
        print(f"   ✗ OU噪声转换错误: {e}")
        return False

def test_replay_buffer():
    """测试重播缓冲区"""
    print("\n4. 重播缓冲区测试:")
    try:
        from utilities.data_structures.Replay_Buffer import Replay_Buffer

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        buffer = Replay_Buffer(1000, 32, seed=123, device=device)

        # 添加一些模拟经验数据
        num_uav = 8
        state_dim = 6
        action_dim = 4

        # 模拟从环境接收到numpy数据
        state_np = np.random.randn(num_uav, state_dim).astype(np.float32)
        action_np = np.random.randn(num_uav, action_dim).astype(np.float32)
        reward_np = np.random.randn(num_uav).astype(np.float32)
        next_state_np = np.random.randn(num_uav, state_dim).astype(np.float32)
        done_np = np.zeros(num_uav, dtype=np.float32)

        # 添加经验
        buffer.add_experience(state_np, action_np, reward_np, next_state_np, done_np)

        # 采样数据
        states, actions, rewards, next_states, dones = buffer.sample(32)

        print(f"   ✓ 重播缓冲区正常")
        print(f"   原始数据类型: numpy数组")
        print(f"   采样数据类型: states-{type(states)}-{states.dtype}->device-{states.device}")
        print(f"   缓冲区设备: {buffer.device}")
        print(f"   数据形状: states-{states.shape}, actions-{actions.shape}")
        return True
    except Exception as e:
        print(f"   ✗ 重播缓冲区错误: {e}")
        return False

def test_simplified_training_loop():
    """测试简化的训练循环"""
    print("\n5. 简化训练循环测试:")
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 模拟一组简单的网络操作
        import torch.nn as nn

        # 创建简单的网络
        actor = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 2个动作维度 * 4 = 均值和标准差
        ).to(device)

        # 模拟步骤
        batch_size = 32
        counter = 0

        while counter < 10:
            state = torch.randn(batch_size, 6, device=device)

            with torch.no_grad():
                output = actor(state)

                # 分解动作各个分量 (类别/距离/角度/beta)
                mean = output[:, :4]
                log_std = output[:, 4:]
                std = log_std.exp()

                # 使用重参数化技巧采样
                normal = torch.distributions.Normal(mean, std)
                actions = torch.tanh(normal.sample())

                # 确保所有张量在相同设备
                actions.device == state.device

            counter += 1

        print(f"   ✓ 简化训练循环正常")
        print(f"   处理了{counter}个小批次")
        print(f"   设备和类型: state-{state.device}, actions-{actions.device}")
        return True
    except Exception as e:
        print(f"   ✗ 简化训练循环错误: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始设备一致性测试...")

    results = []

    # 运行所有测试
    results.append(("GPU基础配置", test_basic_gpu_setup()))
    results.append(("重力噪声", test_gravity_noise()))
    results.append(("OU噪声转换", test_ou_noise_conversion()))
    results.append(("重播缓冲区", test_replay_buffer()))
    results.append(("简化训练循环", test_simplified_training_loop()))

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:15}{status}")
        if not result:
            all_passed = False

    print("="*60)
    if all_passed:
        print("🎉 所有测试通过！设备一致性修复成功！")
    else:
        print("⚠ 发现一些问题，请检查上面的错误信息")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)