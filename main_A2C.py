import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import openseespy.opensees as ops

# ================== 超参数区（可根据需要自行修改） ==================
NUM_EPISODES = 1000  # 训练总轮数
NUM_FLOORS = 10  # 楼层数
LEARNING_RATE = 0.005  # 学习率
UPDATE_FREQUENCY = 30  # 每隔多少轮执行一次模型更新
FLOOR_HEIGHT = 3.0
MASS_PER_FLOOR = 1e5
K_FLOOR = 1e8
DT = 0.02
ACCEL_FILE = 'accel.txt'


# =============== 1. Actor-Critic 网络：用于A2C ===============
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_floors):
        super(ActorCriticNetwork, self).__init__()

        # 共有部分
        # 注意：示例里输入只有 1 维(state = [[0.0]])，实际可改为更复杂形态
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)

        # actor 分支：输出动作概率分布
        self.actor_out = nn.Linear(128, num_floors)

        # critic 分支：输出状态价值 V(s)
        self.critic_out = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (batch_size, 1)，表示状态向量
        返回：
          action_probs: (batch_size, num_floors)，策略网络给出的动作概率
          state_value:  (batch_size, 1)，价值网络给出的 V(s)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # actor 分支
        action_logits = self.actor_out(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # critic 分支
        state_value = self.critic_out(x)  # 不用softmax，直接输出一个标量
        return action_probs, state_value


# ================== 环境定义：BuildingEnv ==================
class BuildingEnv:
    def __init__(self,
                 num_floors=NUM_FLOORS,
                 floor_height=FLOOR_HEIGHT,
                 mass_per_floor=MASS_PER_FLOOR,
                 k_floor=K_FLOOR,
                 dt=DT,
                 accel_file=ACCEL_FILE):

        self.num_floors = num_floors
        self.floor_height = floor_height
        self.mass_per_floor = mass_per_floor
        self.k_floor = k_floor
        self.dt = dt

        # 读取地震波
        with open(accel_file, 'r') as f:
            self.acc_values = [float(line.strip()) for line in f]
        self.num_steps = len(self.acc_values)

    def step(self, action):
        """
        action: 整数，表示选择在哪一层（1~num_floors）添加一个额外的twoNodeLink(阻尼器)
        返回：
          alpha_value (float), done (bool)
        """
        alpha_value = self.run_opensees_analysis(action)
        done = True  # 本示例中，每次step后就结束
        return alpha_value, done

    def run_opensees_analysis(self, damper_floor):
        """
        使用给定的 damper_floor，在对应节点间加阻尼器并做时程分析。
        返回：alpha_value
        """
        # 清空OpenSees模型
        ops.wipe()

        # 创建基本模型
        ops.model('basic', '-ndm', 1, '-ndf', 1)

        # 创建节点
        for i in range(self.num_floors + 1):
            ops.node(i + 1, i * self.floor_height)
        # 固定底部节点
        ops.fix(1, 1)

        # 定义材料
        matTag = 1
        ops.uniaxialMaterial('Elastic', matTag, self.k_floor)

        # 定义楼层质量
        for i in range(2, self.num_floors + 2):
            ops.mass(i, self.mass_per_floor)

        # 创建楼层之间的twoNodeLink
        elementTag = 1
        for i in range(1, self.num_floors + 1):
            ops.element('twoNodeLink', elementTag, i, i + 1, '-mat', matTag, '-dir', 1)
            elementTag += 1

        # 在 action 所对应的楼层，加一个额外阻尼器
        ops.element('twoNodeLink', 9999, damper_floor, damper_floor + 1, '-mat', matTag, '-dir', 1)

        # 定义地震动
        timeSeriesTag = 1
        ops.timeSeries('Path', timeSeriesTag, '-dt', self.dt, '-values', *self.acc_values)
        patternTag = 1
        ops.pattern('UniformExcitation', patternTag, 1, '-accel', timeSeriesTag)

        # 瑞利阻尼(2%)，此处示例写法
        zeta = 0.02
        betaK = 2 * zeta / (1.0 / (2 * 3.14159) + 1.0 / (2 * 3.14159))
        ops.rayleigh(0.0, betaK, 0.0, 0.0)

        # 求解器设置
        ops.system('BandGeneral')
        ops.numberer('Plain')
        ops.constraints('Plain')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.algorithm('Newton')
        ops.analysis('Transient')

        drifts = []
        for step in range(self.num_steps):
            ok = ops.analyze(1, self.dt)
            if ok != 0:
                # 如果分析失败，直接返回0或其他标记
                return 0.0
            # 计算位移
            displacements = [ops.nodeDisp(floor, 1) for floor in range(1, self.num_floors + 2)]
            # 计算层间位移角
            for i in range(self.num_floors):
                drift = (displacements[i + 1] - displacements[i]) / self.floor_height
                drifts.append(drift)

        # 如果任意层间位移角 > 1/200，则 alpha_value=0
        for d in drifts:
            if abs(d) > 1.0 / 200.0:
                return 0.0

        # 否则 alpha_value = sum(|drift_i|)
        alpha_value = sum(abs(d) for d in drifts)
        # 额外加上出现过的最大层间位移角 * 3000
        max_drift = max(abs(d) for d in drifts)
        alpha_value += max_drift * 3000

        return alpha_value


# ================== 2. 使用A2C进行训练 ==================
def train_a2c(num_episodes=NUM_EPISODES,
              num_floors=NUM_FLOORS,
              lr=LEARNING_RATE,
              update_freq=UPDATE_FREQUENCY):
    env = BuildingEnv(num_floors=num_floors)
    actor_critic = ActorCriticNetwork(num_floors)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    actions_history = []
    rewards_history = []
    floor_rewards = {floor: [] for floor in range(1, num_floors + 1)}

    # 用于batch更新的缓存
    batch_loss = 0.0
    batch_count = 0

    for episode in range(num_episodes):
        # 示例：状态仍然只用 [[0.0]]
        state_tensor = torch.tensor([[0.0]], dtype=torch.float)

        # 前向：获取动作概率分布 & 状态价值
        action_probs, state_value = actor_critic(state_tensor)
        dist = torch.distributions.Categorical(action_probs)

        # 采样动作(楼层)
        action_idx = dist.sample()
        floor_selected = action_idx.item() + 1

        # 与环境交互
        reward, done = env.step(floor_selected)

        # 记录历史
        actions_history.append(floor_selected)
        rewards_history.append(reward)
        floor_rewards[floor_selected].append(reward)

        # 打印每一轮信息
        print(f"Episode {episode + 1}, Action {floor_selected}, Reward {reward:.6f}")

        # -------------- 关键区别：使用 Advantage 来更新--------------
        # Critic 给出的当前状态价值
        value_s = state_value[0, 0]  # shape是(1,1)，取标量
        # Advantage = (真实奖励) - (Critic 估计)
        advantage = reward - value_s.item()

        # actor 的损失
        log_prob = dist.log_prob(action_idx)
        actor_loss = -log_prob * advantage

        # critic 的损失 (MSE)
        value_target = torch.tensor([reward], dtype=torch.float)
        critic_loss = F.mse_loss(state_value.view(-1), value_target)

        # 总损失（可以加上一个权重系数）
        total_loss = actor_loss + 0.5 * critic_loss

        batch_loss += total_loss
        batch_count += 1

        # 如果达到 update_freq 就做一次反向传播和更新
        if (episode + 1) % update_freq == 0:
            optimizer.zero_grad()
            loss_to_optimize = batch_loss / batch_count
            loss_to_optimize.backward()
            optimizer.step()

            # 清空batch
            batch_loss = 0.0
            batch_count = 0

    # 如果最后一批不足update_freq，也需要更新
    if batch_count > 0:
        optimizer.zero_grad()
        loss_to_optimize = batch_loss / batch_count
        loss_to_optimize.backward()
        optimizer.step()

    # ====== 可视化结果 ======
    episodes_range = np.arange(1, num_episodes + 1)
    floor_counts = [len(floor_rewards[f]) for f in range(1, num_floors + 1)]

    floor_avg_rewards = []
    for f in range(1, num_floors + 1):
        if len(floor_rewards[f]) > 0:
            floor_avg = np.mean(floor_rewards[f])
        else:
            floor_avg = 0.0
        floor_avg_rewards.append(floor_avg)

    plt.figure(figsize=(12, 8))

    # (1) 每回合动作分布
    plt.subplot(2, 2, 1)
    plt.title("Action per Episode")
    plt.plot(episodes_range, actions_history, marker='o', markersize=4, linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Selected Floor")

    # (2) 每回合奖励
    plt.subplot(2, 2, 2)
    plt.title("Reward per Episode")
    plt.plot(episodes_range, rewards_history, marker='o', markersize=4, linestyle='--', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Alpha Value")

    # (3) 各楼层被选次数
    plt.subplot(2, 2, 3)
    plt.title("Times Each Floor Selected")
    plt.bar(range(1, num_floors + 1), floor_counts, color='green')
    plt.xlabel("Floor")
    plt.ylabel("Times Chosen")

    # (4) 各楼层平均奖励
    plt.subplot(2, 2, 4)
    plt.title("Average Reward for Each Floor")
    plt.bar(range(1, num_floors + 1), floor_avg_rewards, color='purple')
    plt.xlabel("Floor")
    plt.ylabel("Average Alpha Value")

    plt.tight_layout()
    plt.show()

    print("A2C Training finished. Plots have been shown.")


# ================ 主函数入口运行时调用此函数 ================
if __name__ == "__main__":
    train_a2c()
