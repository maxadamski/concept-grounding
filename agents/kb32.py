import torch
import torch.nn as nn
from env import OBS_DIM, ACTION_DIM, policy_mask
from kb import *

EMB_DIM = 32
CONCEPT_DIM = 19
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
		self.tile_decoder = nn.Linear(EMB_DIM, CONCEPT_DIM)
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
			loss = self.tile_loss(tiles, self.tile_decoder(embeddings))
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

	def tile_loss(self, x, Y):
		# tested with lambda=0.8, lambda=0.95, lambda=0.98
		B, T, C = x.shape
		D = CONCEPT_DIM
		xv = x.reshape(B*T, C)
		y = torch.zeros(B, T, D).to(self.device)
		yv = y.view(B*T, D) 
		yv[:,0] = (xv[:,WATER]>0) | (xv[:,FOUNTAIN]>0) # drink
		yv[:,1] = (xv[:,RIPE_PLANT]>0) | (xv[:,PASSIVE_1]>0) | (xv[:,PASSIVE_2]>0) | (xv[:,PASSIVE_3]>0) # eat
		yv[:,2] = (xv[:,RANGED_1]>0) | (xv[:,RANGED_2]>0) | (xv[:,RANGED_3]>0) # archers
		yv[:,3] = (xv[:,MELEE_1]>0) | (xv[:,MELEE_2]>0) | (xv[:,MELEE_3]>0) # warriors
		yv[:,4] = (yv[:,2]>0) | (yv[:,3]>0) # all enemies
		yv[:,5] = (yv[:,4]>0) | (xv[:,ENEMY_PROJECTILE_1]>0) | (xv[:,ENEMY_PROJECTILE_2]>0) | (xv[:,ENEMY_PROJECTILE_3]>0) # harmful
		yv[:,6] = (xv[:,CRAFTING_TABLE]>0) | (xv[:,FURNACE]>0)
		yv[:,7] = (xv[:,TREE]>0) | (xv[:,WOOD]>0) | (xv[:,PLANT]>0) | (xv[:,CHEST]>0) # no pickaxe
		yv[:,8] = (yv[:,7]>0) | (xv[:,STONE]>0) | (xv[:,COAL]>0) # wood tier
		yv[:,9] = (yv[:,8]>0) | (xv[:,IRON]>0) # stone tier
		yv[:,10] = (yv[:,9]>0) | (xv[:,DIAMOND]>0) # iron tier
		yv[:,11] = (yv[:,10]>0) | (xv[:,RUBY]>0) | (xv[:,SAPPHIRE]>0) # diamond tier
		yv[:,12] = (xv[:,WALL]>0) | (xv[:,WALL_MOSS]>0) | (xv[:,OUT_OF_BOUNDS]>0) # non-removable
		yv[:,13] = (xv[:,WATER]>0) | (xv[:,LAVA]>0) # can place block
		yv[:,14] = (xv[:,LADDER_DOWN]>0)
		yv[:,15] = (xv[:,LADDER_UP]>0)
		yv[:,16] = (xv[:,LADDER_DOWN]>0) | (xv[:,LADDER_DOWN_BLOCKED]>0)
		yv[:,17] = (xv[:,CHEST]>0)
		yv[:,18] = (xv[:,LIGHT_LEVEL]<0.05) # visibility mask
		w = 1/(yv.mean(0) + 1e-8) - 1
		return nn.functional.binary_cross_entropy_with_logits(Y, y, pos_weight=w)
