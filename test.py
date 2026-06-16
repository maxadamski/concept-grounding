import argparse
import torch
import numpy as np

from time import time
from craftax import Action, Achievement
from agent import load_agent_from_save
from env import CraftaxSymbolicRollout, ACTION_DIM, ACHIEVEMENTS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/best.sav")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--parallel", type=int, default=64)
args = parser.parse_args()

EPS = 1e-7
REWARD_SCALE = 1/226

seed = args.seed if args.seed >= 0 else np.random.randint(2**31)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

agent = load_agent_from_save(args.model, parallel=args.parallel, device=args.device)
agent.eval()

envs = CraftaxSymbolicRollout(num_envs=args.parallel, seed=seed)

obs = envs.reset([i for i in range(args.parallel)])
score = np.array([0.0 for _ in range(args.parallel)])
depth = np.array([0 for _ in range(args.parallel)])

total_steps = 0
history = []
terminals = []

start_time = time()
achievements = np.zeros((len(ACHIEVEMENTS),))

with torch.no_grad():
    while len(terminals) < args.episodes:
        obs = torch.Tensor(obs).to(args.device)
        belief = agent.belief()
        state = agent.state(obs, belief)
        policy = agent.mask(agent.actor(state), obs).softmax(-1)
        #action = torch.argmax(policy, -1).reshape(-1)
        action = torch.multinomial(policy, 1).reshape(-1)
        obs, reward, terminal, timeout, info = envs.step(action.cpu().numpy())
        if np.any(terminal):
            achievements += info['achievements'][terminal].sum(0)
        total_steps += args.parallel
        score += reward
        depth += 1
        history += list(zip(depth, score))
        terminals += list(zip(depth[terminal], score[terminal]))
        #if np.any(terminal): print(score[terminal])
        depth[terminal] = 0
        score[terminal] = 0.0

scores = [s*REWARD_SCALE for d, s in terminals]
depths = [d for d, s in terminals]

achievements = zip(achievements/len(terminals), ACHIEVEMENTS)
probs = {name: prob for prob, name in achievements}

results = [
    ('elapsed', time() - start_time),
    ('reward/count', len(scores)),
    ('reward/mean', np.mean(scores)),
    ('reward/std', np.std(scores)),
    ('reward/max', max(scores)),
    ('depth/mean', np.mean(depths)),
    ('depth/std', np.std(depths)),
    ('P(open_chest|enter_dungeon)', probs['open_chest']/(probs['enter_dungeon']+EPS)),
    ('P(eat_snail|enter_dungeon)', probs['eat_snail']/(probs['enter_dungeon']+EPS)),
    ('P(defeat_orc_soldier|enter_dungeon)', probs['defeat_orc_solider']/(probs['enter_dungeon']+EPS)),
    ('P(defeat_orc_mage|enter_dungeon)', probs['defeat_orc_mage']/(probs['enter_dungeon']+EPS)),
    ('P(enter_gnomish_mines|enter_dungeon)', probs['enter_gnomish_mines']/(probs['enter_dungeon']+EPS)),
    ('P(eat_bat|enter_gnomish_mines)', probs['eat_bat']/(probs['enter_gnomish_mines']+EPS)),
    ('P(defeat_gnome_warrior|enter_gnomish_mines)', probs['defeat_gnome_warrior']/(probs['enter_gnomish_mines']+EPS)),
    ('P(defeat_gnome_archer|enter_gnomish_mines)', probs['defeat_gnome_archer']/(probs['enter_gnomish_mines']+EPS)),
]

results += [(f'P({name})', prob) for name, prob in probs.items()]

print(args.model)
for name, prob in results:
    print(f'{prob:.6f} {name}')
print()

import matplotlib.pyplot as plt
#depth, score  = np.array(terminals).T
#plt.hist2d(depth, score, bins=20)
plt.hist(scores, bins=16)
plt.xlabel('score')
plt.ylabel('count')
plt.show()
