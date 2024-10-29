import torch
import torch.optim as optim
import numpy as np
from utils import rollout
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32):
    
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    for epoch in range(num_epochs): 
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            #========== TODO: start ==========
            # Fill in your behavior cloning implementation here

            # Sample a batch of expert data
            batch_idxs = np.random.choice(idxs, batch_size)
            states = torch.tensor([expert_data[idx]['observations'] for idx in batch_idxs]).float()
            actions = torch.tensor([expert_data[idx]['actions'] for idx in batch_idxs]).float()

            # Compute the log-likelihood of the expert actions
            log_likelihood = policy.log_prob(states, actions)

            # Define the loss as the negative log-likelihood
            loss = -log_likelihood.mean()

            #========== TODO: end ==========
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / num_batches))
        losses.append(loss.item())
    with open("losses.json", "w") as f:
        json.dump(losses, f)