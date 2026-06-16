import torch
import torch.nn as nn
from env import OBS_DIM, ACTION_DIM, policy_mask

STATE_DIM = OBS_DIM
BELIEF_DIM = 0
NN_WIDTH = 512

class Agent(nn.Module):
	def __init__(self, *, parallel, device):
		super().__init__()
		self.parallel = parallel
		self.device = device
		self.state_dim = STATE_DIM
		self.belief_dim = BELIEF_DIM
		self.actor = nn.Sequential(
			nn.Linear(STATE_DIM, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, ACTION_DIM),
		)
		self.critic = nn.Sequential(
			nn.Linear(STATE_DIM, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, NN_WIDTH),
			nn.ReLU(),
			nn.Linear(NN_WIDTH, 1),
		)

	def state(self, obs, belief, return_attention=False, with_loss=False):
		result = obs
		if with_loss:
			loss = torch.Tensor(1).to(self.device)
			return result, loss
		if return_attention:
			return result, None
		return result

	def update(self, obs, action, reward, terminal, timeout, info, next_obs):
		pass

	def belief(self):
		return torch.zeros(self.parallel, BELIEF_DIM)

	def policy(self, obs):
		return self.mask(self.actor(obs), obs).softmax(-1)

	def mask(self, policy, obs):
		return policy_mask(obs, policy)
