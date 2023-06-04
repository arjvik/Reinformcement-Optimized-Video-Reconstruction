import torch

def compute_rewards_to_go(rewards, gamma=1):
    b, t = rewards.size()  # Get the batch size and sequence length
    rewards_to_go = torch.zeros_like(rewards)  # Initialize a tensor of rewards-to-go with the same shape as rewards
    
    for i in range(b):  # Iterate through each batch
        reward_to_go = 0  # Initialize the reward-to-go for the last step as 0
        for j in reversed(range(t)):  # Iterate through the rollout in reverse order (from the last step to the first step)
            reward_to_go = rewards[i, j] + gamma * reward_to_go  # Calculate the reward-to-go for the current step
            rewards_to_go[i, j] = reward_to_go  # Store the calculated reward-to-go in the rewards_to_go tensor
    
    return rewards_to_go


##### TODO
### First, need to have access to rollouts as described below
# reward = [] # marginal reward at each step, this is just from -(LPIPS(target, pred) - LPIPS(prev, pred))
# reward[-1] += "spatio-temporal loss"
# rollout = ("batch of states", "batch of actions", "batch of log probabilities", compute_rewards_to_go(reward))
### Second, both policy networks need to have a .get_log_prob(state, action) 
### method that takes in a state, computes softmax, and then returns log prob of the action given that softmax

def ppo(self, actor, critic, actor_optim, critic_optim, rollout, n_updates_per_iteration=5):                                                                
        i_so_far += 1
        obs, acs, log_prob, rtgs = rollout                  
        V = critic(obs)
        A_k = rtgs - V.detach()                                                                  
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        logger = {
             'actor_losses': [],
        }

        for _ in range(n_updates_per_iteration):                                                    
            V = critic(obs)
            curr_log_prob = actor.get_log_prob(obs, acs)
            ratio = torch.exp(curr_log_prob - log_prob)

            L1 = ratio * A_k
            L2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = -torch.min(L1, L2).mean()
            critic_loss = torch.nn.MSELoss()(V, rtgs)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            logger['actor_losses'].append(actor_loss.detach())
