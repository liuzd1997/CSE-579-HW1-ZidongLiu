import torch
import numpy as np
import json
import random

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

        # Flatten the expert data subset for training
        flattened_expert = {'observations': [], 'actions': []}
        for expert_path in sampled_expert_data:
            for k in flattened_expert.keys():
                flattened_expert[k].append(expert_path[k])
        for k in flattened_expert.keys():
            flattened_expert[k] = np.concatenate(flattened_expert[k])

        # Initialize optimizer
        optimizer = torch.optim.Adam(list(policy.parameters()), lr=1e-4)
        num_batches = len(flattened_expert['observations']) // batch_size

        # Train the policy with the current data subset
        for epoch in range(num_epochs):
            indices = np.arange(len(flattened_expert['observations']))
            np.random.shuffle(indices)
            running_loss = 0.0
            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                batch_states = torch.tensor(flattened_expert['observations'][batch_indices]).float().to(device)
                batch_actions = torch.tensor(flattened_expert['actions'][batch_indices]).float().to(device)

                optimizer.zero_grad()
                log_probs = policy.log_prob(batch_states, batch_actions)
                loss = -log_probs.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Print average loss every 10 epochs for monitoring
            if epoch % 10 == 0:
                print(f'[Data Ratio: {ratio * 100:.0f}%] Epoch [{epoch}/{num_epochs}], Loss: {running_loss / num_batches:.4f}')

        # Evaluate the trained policy
        avg_reward = evaluate(env, policy, num_validation_runs=20, episode_length=episode_length)
        performance_scores.append(avg_reward)

    # Save the performance scores for each data ratio
    with open("performance_scores.json", "w") as f:
        json.dump(performance_scores, f)

    return performance_scores