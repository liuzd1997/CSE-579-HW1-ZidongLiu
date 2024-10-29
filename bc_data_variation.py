import torch
import numpy as np
import json
import random
from evaluate import evaluate

def simulate_policy_bc_with_data_variation(env, policy, expert_data, data_ratios=[0.1, 0.3, 0.5, 0.7, 1.0],
                                           num_epochs=500, episode_length=50, batch_size=32):
    """
    Train the BC agent using varying amounts of expert data and record performance.

    Args:
        env: The environment in which the policy will be trained.
        policy: The policy model to be trained.
        expert_data: Full expert data used to sample subsets for training.
        data_ratios: List of data proportions (e.g., [0.1, 0.3, 0.5, 1.0]) to investigate.
        num_epochs: Number of epochs for training each subset.
        episode_length: The length of each episode in the environment.
        batch_size: Batch size used for training.

    Returns:
        performance_scores: List of average performance scores for each data ratio.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    performance_scores = []

    for ratio in data_ratios:
        # Sample a subset of the expert data based on the current ratio
        sample_size = int(ratio * len(expert_data))
        sampled_expert_data = random.sample(expert_data, sample_size)
        idxs = np.array(range(len(sampled_expert_data)))
        num_batches = len(idxs)*episode_length // batch_size
        losses = []

        # Initialize optimizer
        optimizer = torch.optim.Adam(list(policy.parameters()), lr=1e-4)
      

        # Train the policy with the current data subset
        for epoch in range(num_epochs):
          np.random.shuffle(idxs)
          running_loss = 0.0
          for i in range(num_batches):
            optimizer.zero_grad()
            t1_idx = np.random.randint(len(sampled_expert_data), size=(batch_size,)) # Indices of first trajectory
            t1_idx_pertraj = [np.random.randint(sampled_expert_data[c_idx]['observations'].shape[0]) for c_idx in t1_idx]
            t1_states = np.concatenate([sampled_expert_data[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([sampled_expert_data[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])

            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)
            #========== TODO: start ==========
            # Fill in your behavior cloning implementation here
            log_probs = policy.log_prob(t1_states, t1_actions)
            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print average loss every 10 epochs for monitoring
            if epoch % 10 == 0:
                print(f'[Data Ratio: {ratio * 100:.0f}%] Epoch [{epoch}/{num_epochs}], Loss: {running_loss / num_batches:.4f}')

        # Evaluate the trained policy
        Success_rate = evaluate(env, policy,'behavior_cloning', num_validation_runs=20, episode_length=episode_length,env_name='reacher')
        print('avg_reward=',Success_rate)
        performance_scores.append(Success_rate)

    # Save the performance scores for each data ratio
    with open("performance_scores.json", "w") as f:
        json.dump(performance_scores, f)

    return performance_scores