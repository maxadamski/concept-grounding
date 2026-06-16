import torch
import numpy as np
import jax
import jax.numpy as jnp

from craftax.craftax.constants import Action, Achievement
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world
from craftax.wrappers import OptimisticResetVecEnvWrapper
from craftax import make_craftax_env_from_name

ACTION_DIM = len(Action)
OBS_DIM = 8268
STATE_DIM = 8268
ACHIEVEMENTS = [Achievement(i).name.lower() for i in range(len(Achievement))]
DEFAULT_PARAMS = EnvParams()
STATIC_PARAMS = StaticEnvParams()
MAP_SIZE = STATIC_PARAMS.map_size

def craftax_step_pipeline(rng, state, action, param):
	state, reward = craftax_step(rng, state, action, param, STATIC_PARAMS)
	terminal = is_game_over(state, param, STATIC_PARAMS)
	obs = render_craftax_symbolic(state)
	return state, obs, reward, terminal

craftax_step_jit = jax.jit(craftax_step_pipeline)
craftax_render_symbolic_jit = jax.jit(render_craftax_symbolic)
craftax_vector_step_jit = jax.jit(jax.vmap(craftax_step_pipeline))

def craftax_step_pipeline2(rng, state, action, param, timeout, init_state):
	state, reward = craftax_step(rng, state, action, param, STATIC_PARAMS)
	terminal = is_game_over(state, param, STATIC_PARAMS)
	obs = render_craftax_symbolic(state)
	state = jax.lax.cond(terminal | timeout, lambda _: init_state, lambda _: state, operand=None)
	return state, obs, reward, terminal

craftax_vector_step3_jit = jax.jit(jax.vmap(craftax_step_pipeline2))

def policy_mask(obs, policy):
	if len(obs.shape) == 1:
		obs = obs.reshape(1, -1)
	if len(policy.shape) == 1:
		policy = policy.reshape(1, -1)

	mask = -1e8
	sleeping = obs[:,8261] > 0

	no_wood = obs[:,8217] < 0.1
	no_sapling = obs[:,8224] < 0.1
	no_stone = obs[:,8218] < 0.1
	no_torches = obs[:,8225] < 0.1
	no_arrows = obs[:,8226] < 0.1
	no_hole = obs[:,(4*11+5)*83 + 39] < 1
	no_ladder = obs[:,(4*11+5)*83 + 40] < 1
	no_table = ~(
		(obs[:,(3*11+4)*83 + 11] > 0) |
		(obs[:,(4*11+4)*83 + 11] > 0) |
		(obs[:,(5*11+4)*83 + 11] > 0) |
		(obs[:,(3*11+5)*83 + 11] > 0) |
		(obs[:,(5*11+5)*83 + 11] > 0) |
		(obs[:,(3*11+6)*83 + 11] > 0) |
		(obs[:,(4*11+6)*83 + 11] > 0) |
		(obs[:,(5*11+6)*83 + 11] > 0)
	)
	no_table_and_furnace = no_table & ~(
		(obs[:,(3*11+4)*83 + 12] > 0) |
		(obs[:,(4*11+4)*83 + 12] > 0) |
		(obs[:,(5*11+4)*83 + 12] > 0) |
		(obs[:,(3*11+5)*83 + 12] > 0) |
		(obs[:,(5*11+5)*83 + 12] > 0) |
		(obs[:,(3*11+6)*83 + 12] > 0) |
		(obs[:,(4*11+6)*83 + 12] > 0) |
		(obs[:,(5*11+6)*83 + 12] > 0)
	)

	policy[~sleeping,0] = mask # noop
	policy[sleeping,1:] = mask # everything else
	policy[no_stone,7] = mask # place_stone
	policy[no_wood,8] = mask # place_table
	policy[no_stone,9] = mask # place_furnace
	policy[no_sapling,10] = mask # place_plant
	policy[no_table,11] = mask # make wood pickaxe
	policy[no_table,12] = mask # make stone pickaxe
	policy[no_table_and_furnace,13] = mask # make iron sword
	policy[no_table,14] = mask # make wood sword
	policy[no_table,15] = mask # make stone sword
	policy[no_table_and_furnace,16] = mask # make iron sword
	policy[:,17] = mask # rest is useless? :)
	policy[no_hole,18] = mask # descend
	policy[no_ladder,19] = mask # ascend
	policy[no_table,20] = mask # make diamond pickaxe
	policy[no_table,21] = mask # make diamond pickaxe
	policy[no_table_and_furnace,22] = mask # make iron armor
	policy[no_table,23] = mask # make diamond armor
	policy[no_arrows,24] = mask # shoot arrow
	policy[no_table,25] = mask # make arrow
	policy[:,26:28] = mask # cast fire/iceball
	policy[no_torches,28] = mask # place torch
	policy[:,29:38] = mask # drink potion, read book, enchant sword/armor
	policy[no_table,38] = mask # make torch
	policy[:,39:42] = mask # level up
	policy[:,42] = mask # enchant bow

	return policy

class CraftaxSymbolic:
	def __init__(self):
		self.params = EnvParams()

	def reset(self, seed):
		self._rng = jax.random.key(seed)
		self.state = generate_world(jax.random.key(seed), self.params, STATIC_PARAMS)
		return self.observe()

	def step(self, action):
		rng, self._rng = jax.random.split(self._rng)
		info = {'achievements': np.array(self.state.achievements)}
		self.state, obs, reward, terminal = craftax_step_jit(rng, self.state, int(action), self.params)
		return np.array(obs), np.array(reward), np.array(terminal), info

	def observe(self):
		obs = craftax_render_symbolic_jit(self.state)
		return np.array(obs)

class CraftaxSymbolicParallel:
	action_dim = ACTION_DIM
	state_dim = STATE_DIM

	def __init__(self, num_envs, seed, reset_ratio=16, gpu=False):
		self.num_envs = num_envs
		self._rng = jax.random.PRNGKey(seed)
		self.env = make_craftax_env_from_name('Craftax-Symbolic-v1', auto_reset=False)
		self.env = OptimisticResetVecEnvWrapper(self.env, num_envs=num_envs, reset_ratio=16)
		self.env_params = self.env.default_params

	def pregenerate(self):
		pass

	def reset(self):
		rng, self._rng = jax.random.split(self._rng)
		obs, self.env_state = self.env.reset(rng, self.env_params)
		return np.array(obs)

	def step(self, action):
		action = jnp.array(action)
		achievements = np.array(self.env_state.achievements)
		rng, self._rng = jax.random.split(self._rng)
		next_obs, self.env_state, reward, terminal, _ = self.env.step(rng, self.env_state, action, self.env_params)
		terminal = np.array(terminal)
		return np.array(next_obs), np.array(reward), terminal, terminal, {'achievements': achievements}

class CraftaxSymbolicRollout:
	action_dim = ACTION_DIM
	state_dim = STATE_DIM

	def __init__(self, seed, num_envs, timeout=None):
		self.num_envs = num_envs
		self.params = jax.tree.map(lambda *x: jnp.stack(x), *[DEFAULT_PARAMS for _ in range(num_envs)])
		self.rng = jax.random.split(jax.random.key(seed), num_envs)
		self.time = np.zeros((num_envs,))
		self.timeout = timeout or 100_000
		self.fold = 0

	def reset(self, seed):
		rngs = jnp.array([jax.random.key(x) for x in seed])
		self.state = self.init_state = jax.vmap(lambda rng: generate_world(rng, DEFAULT_PARAMS, STATIC_PARAMS))(rngs)
		obs = jax.vmap(craftax_render_symbolic_jit)(self.state)
		return np.array(obs)

	def step(self, action):
		rng = jax.vmap(lambda rng: jax.random.fold_in(rng, self.fold))(self.rng)
		self.fold += 1
		achievements = np.array(self.state.achievements)
		timeout = self.time > self.timeout
		self.state, obs, reward, terminal = craftax_vector_step3_jit(
			rng, self.state, action, self.params, timeout, self.init_state)
		self.time += 1
		self.time[timeout | terminal] = 0
		return np.array(obs), np.array(reward), np.array(terminal), timeout, {'achievements': achievements}
