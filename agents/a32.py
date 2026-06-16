import torch
import torch.nn as nn
from env import OBS_DIM, ACTION_DIM, policy_mask
from kb import *

EMB_DIM = 32
STATE_DIM = EMB_DIM*99 + STATUS_DIM
BELIEF_DIM = 0
NN_WIDTH = 512

class Agent(nn.Module):
	def __init__(self, *, device, parallel):
		super().__init__()
		self.device = device
		self.parallel = parallel
		self.state_dim = STATE_DIM
		self.belief_dim = BELIEF_DIM
		self.tile_encoder = nn.Sequential(
			nn.Linear(TILE_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, EMB_DIM),
		)
		self.tile_decoder = nn.Sequential(
			nn.Linear(EMB_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, TILE_DIM),
			nn.ReLU(),
			nn.Linear(TILE_DIM, TILE_DIM),
		)
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
		batch = obs.size(0)
		tiles = obs[:,:8217].view(batch, 99, TILE_DIM)
		status = obs[:,8217:].view(batch, STATUS_DIM)
		embeddings = self.tile_encoder(tiles)
		flat = embeddings.view(batch, 99*EMB_DIM)
		result = torch.cat([flat, status], axis=1)
		if with_loss:
			loss = self.autoencoder_loss(tiles, self.tile_decoder(embeddings))
			return result, loss
		if return_attention:
			return result, None
		return result

	def autoencoder_loss(self, x, Y):
	    y = x
	    B, T, C = x.shape
	    m = y.reshape(B*T, C).mean(0)
	    w = 1/(m + 1e-8) - 1
	    return torch.nn.functional.binary_cross_entropy_with_logits(Y, y, pos_weight=w)

	def update(self, obs, action, reward, terminal, timeout, info, next_obs):
		pass

	def belief(self):
		return torch.zeros(self.parallel, BELIEF_DIM)

	def policy(self, obs):
		return self.mask(self.actor(obs), obs).softmax(-1)

	def mask(self, policy, obs):
		return policy_mask(obs, policy)
