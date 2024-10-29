import torch
import torch.optim as optim
import numpy as np
from utils import rollout
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                      batch_size=32):
    
    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    
    # 预处理专家数据，将轨迹展平
    flat_observations = []
    flat_actions = []
    
    for trajectory in expert_data:
        obs = trajectory['observations']
        acts = trajectory['actions']
        # 确保observations和actions长度匹配
        min_len = min(len(obs), len(acts))
        flat_observations.extend(obs[:min_len])
        flat_actions.extend(acts[:min_len])
    
    # 转换为numpy数组以便于处理
    flat_observations = np.array(flat_observations)
    flat_actions = np.array(flat_actions)
    
    # 计算数据集大小和批次数
    dataset_size = len(flat_observations)
    num_batches = dataset_size // batch_size
    
    # 创建索引数组
    idxs = np.array(range(dataset_size))
    losses = []
    
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        
        for i in range(num_batches):
            optimizer.zero_grad()
            
            # 采样批次数据
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_idxs = idxs[batch_start:batch_end]
            
            # 获取批次的观察和动作
            states = torch.tensor(flat_observations[batch_idxs]).float().to(device)
            actions = torch.tensor(flat_actions[batch_idxs]).float().to(device)
            
            # 计算动作的对数似然
            log_likelihood = policy.log_prob(states, actions)
            
            # 定义损失为负对数似然的均值
            loss = -log_likelihood.mean()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / num_batches
        print('[%d] loss: %.8f' % (epoch, epoch_loss))
        losses.append(epoch_loss)
    
    # 保存训练损失历史
    with open("losses.json", "w") as f:
        json.dump(losses, f)