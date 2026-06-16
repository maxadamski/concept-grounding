import argparse
import torch
import torch.nn as nn
import numpy as np
from time import time

from agent import make_agent_from_source, load_agent_from_save
from env import CraftaxSymbolicParallel, OBS_DIM, ACTION_DIM, ACHIEVEMENTS

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--agent', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--save_fork', action='store_true')
parser.add_argument('--ground', type=float, default=0.0)
parser.add_argument('--ground_until', type=float, default=1.0)
args = parser.parse_args()

save_path = args.model
save_freq = args.save_freq
save_fork = args.save_fork
agent_path = args.agent
train_steps = 500_000_000
strict_load = True
debug_loss = True
seed = args.seed

# Updates
learning_rate = 2e-4
num_envs = 1024
rollout_len = 64
epochs = 3
minibatch_size = 4096

# Tuning
gamma = 0.99
gae_lambda = 0.8
clip_grad_norm = 1
clip_eps = 0.2
entropy_bonus = 0.01
ground_lambda = args.ground
ground_steps = train_steps*args.ground_until

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else device

try:
	agent = load_agent_from_save(save_path, parallel=num_envs, device=device)
except FileNotFoundError as error:
	agent = make_agent_from_source(agent_path, parallel=num_envs, device=device)
except Exception as error:
	print('BAD SAVE', save_path)
	raise error

optimizer = torch.optim.AdamW(agent.parameters(), lr=learning_rate)
state_dim = agent.state_dim
belief_dim = agent.belief_dim
obs_dim = OBS_DIM
action_dim = ACTION_DIM

train_steps -= agent.step_count 
batch_size = num_envs*rollout_len
batches = train_steps // batch_size
minibatches = batch_size // minibatch_size
batch_updates = epochs*minibatches
total_updates = batches*batch_updates

print('INFO')
print('  seed         ', seed)
print('  save freq    ', save_freq)
print('CRAFTAX')
print('  obs dim      ', obs_dim)
print('  action dim   ', action_dim)
print('AGENT')
print('  source path  ', agent.source_path)
print('  save path    ', save_path)
print('  param count  ', f'{sum(x.numel() for x in agent.parameters())}')
print('  step count   ', f'{agent.step_count/1e6:.2f}M' if agent.step_count else '0')
print('  state dim    ', state_dim)
print('  belief dim   ', belief_dim)
print('PPO')
print('  train steps  ', f'{train_steps/1e6:.2f}M')
print('  batches      ', batches)
print('  batch size   ', batch_size)
print('  epochs       ', epochs)
print('  minibatch    ', minibatch_size)
print('  updates      ', f'{total_updates/1e3:.2f}K')
print('  updates/batch', batch_updates)
print('  gae lambda   ', f'{gae_lambda:.2f}')
print('  ground lambda', f'{ground_lambda:.2f}')
print('  ground steps ', f'{ground_steps/1e6:.2f}M')
print()

start_time = time()

envs = CraftaxSymbolicParallel(num_envs=num_envs, seed=seed)
obs = envs.reset()
obs = torch.Tensor(obs).to(device)
belief = agent.belief()
# env should keep track of this
env_reward = np.zeros((num_envs,))
env_step = np.zeros((num_envs,))
episode_rewards = []
episode_steps = []
episode_achievements = []

rollout_obs = torch.zeros((rollout_len+1, num_envs, obs_dim)).to(device)
rollout_belief = torch.zeros((rollout_len+1, num_envs, belief_dim)).to(device)
rollout_state = torch.zeros((rollout_len+1, num_envs, state_dim)).to(device)
rollout_action = torch.zeros((rollout_len, num_envs)).int().to(device)
rollout_logprob = torch.zeros((rollout_len, num_envs)).to(device)
rollout_reward = torch.zeros((rollout_len, num_envs)).to(device)
rollout_terminal = torch.zeros((rollout_len, num_envs)).bool().to(device)

for batch in range(batches):
	batch_time = time()
	finished_episodes = len(episode_rewards)

	agent.eval()

	for t in range(rollout_len):
		with torch.no_grad():
			state = agent.state(obs, belief)
			policy = agent.actor(state)
			action = torch.multinomial(agent.mask(policy, obs).softmax(-1), 1).view(num_envs)
			logprob = policy.log_softmax(-1).gather(-1, action.view(num_envs, 1)).view(num_envs)

		next_obs, reward, terminal, timeout, info = envs.step(action.cpu().numpy())
		agent.update(obs, action, reward, terminal, timeout, info, next_obs)

		rollout_obs[t] = obs
		rollout_belief[t] = belief
		rollout_state[t] = state
		rollout_action[t] = action
		rollout_logprob[t] = logprob
		rollout_reward[t] = torch.Tensor(reward).to(device)
		rollout_terminal[t] = torch.Tensor(terminal).bool().to(device)

		belief = agent.belief()
		obs = torch.Tensor(next_obs).to(device)

		done = terminal | timeout
		env_reward += reward
		env_step += 1
		agent.step_count += num_envs
		episode_rewards += list(env_reward[done])
		episode_steps += list(env_step[done])
		episode_achievements += list(info['achievements'][done])
		env_reward[done] = 0
		env_step[done] = 0

	with torch.no_grad():
		rollout_obs[t+1] = torch.Tensor(obs).to(device)
		rollout_belief[t+1] = torch.Tensor(belief).to(device)
		rollout_state[t+1] = agent.state(obs, belief)

		state_view = rollout_state.view((rollout_len+1)*num_envs, state_dim)
		rollout_value = agent.critic(state_view).view(rollout_len+1, num_envs)

		next_gae = 0
		rollout_return = torch.zeros(rollout_reward.shape).to(device)
		for t in reversed(range(rollout_len)):
			delta = rollout_reward[t] + gamma * rollout_value[t+1] * ~rollout_terminal[t] - rollout_value[t]
			next_gae = delta + gamma * gae_lambda * next_gae * ~rollout_terminal[t]
			rollout_return[t] = next_gae + rollout_value[t]

	batch_obs = rollout_obs[:-1].view(batch_size, obs_dim)
	batch_belief = rollout_belief[:-1].view(batch_size, belief_dim)
	batch_value = rollout_value[:-1].view(batch_size)
	batch_action = rollout_action.view(batch_size, 1)
	batch_logprob = rollout_logprob.view(batch_size)
	batch_return = rollout_return.view(batch_size)

	agent.train()

	for epoch in range(epochs):
		permutation = torch.randperm(batch_size).view(minibatches, minibatch_size)
		losses = []

		for minibatch in permutation:
			state, state_loss = agent.state(batch_obs[minibatch], batch_belief[minibatch], with_loss=True)
			value = agent.critic(state).view(minibatch_size)
			advantage = batch_return[minibatch] - value
			value_loss = 0.5*advantage.pow(2).mean()

			policy = agent.actor(state)
			policy_entropy = torch.distributions.Categorical(logits=policy).entropy().mean()

			old_logprob = batch_logprob[minibatch]
			new_logprob = policy.log_softmax(-1).gather(-1, batch_action[minibatch]).view(minibatch_size)
			prob_ratio = torch.exp(new_logprob - old_logprob)
			policy_loss = -prob_ratio * advantage.detach()
			policy_loss_clipped = -prob_ratio.clamp(1-clip_eps, 1+clip_eps) * advantage.detach()
			policy_loss = torch.max(policy_loss, policy_loss_clipped).mean()

			ground_on = 0 if agent.step_count > ground_steps else 1
			loss = policy_loss + value_loss + ground_on*ground_lambda*state_loss - entropy_bonus*policy_entropy
			if debug_loss:
				losses.append((policy_loss.item(), value_loss.item(), state_loss.item()))

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(agent.parameters(), clip_grad_norm)
			optimizer.step()

	window = len(episode_rewards) - finished_episodes
	mean_reward = np.mean(episode_rewards[-window:]) if window else np.nan
	std_reward = np.std(episode_rewards[-window:]) if window else np.nan
	mean_dur = int(np.mean(episode_steps[-window:])) if window else np.nan
	mean_achv = np.mean(episode_achievements[-window:], axis=0) if window else np.zeros(len(ACHIEVEMENTS))
	achievements = {name: prob for name, prob in zip(ACHIEVEMENTS, mean_achv)}
	achievements = sorted(achievements.items(), key=lambda x: -x[1])
	batch_time = time() - batch_time
	total_time = time() - start_time
	sps = int(num_envs*rollout_len / batch_time)
	agent.log.append((agent.step_count, mean_reward, std_reward))
	print(f'STEP {agent.step_count/1e6:.2f}M BATCH {batch+1}/{batches} SPS {sps} EP {len(episode_rewards)} ' + \
		f'REWARD {mean_reward:.2f} ({100*mean_reward/226:.2f}%) STD {std_reward:.2f} STEPS {mean_dur} TIME {batch_time:.1f}s TOTAL {total_time/60:.1f}min')
	if debug_loss:
		losses = np.array(losses)
		policy_loss, value_loss, state_loss = losses.mean(axis=0)
		print(f'L_POLICY {policy_loss:.4f} L_VALUE {value_loss:.4f} L_STATE {state_loss:.6f}')

	for name, prob in achievements:
		if prob > 0:
			print(f"{100*prob:.2f} {name}")

	if batch % save_freq == 0 and batch > 0 or batch == batches - 1:
		save = {'source_path': agent.source_path, 'agent_state': agent.state_dict(),
				'step_count': agent.step_count, 'log': agent.log}

		path = save_path
		if save_fork:
			name, ext = path.split('.')
			path = f'{name}-{int(agent.step_count/1e6)}M.{ext}'

		torch.save(save, path)
		print('SAVED')

	print(flush=True)

print(f'DONE {time() - start_time:.1f}s')
