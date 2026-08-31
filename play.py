import argparse
import os
import sys
import math

from time import time
from datetime import datetime

import torch
import numpy as np
import cv2

from craftax import Action, Achievement
from craftax.screen_renderer import Renderer
from agent import load_agent_from_save
from env import CraftaxSymbolic, ACHIEVEMENTS

BLACK = 0,0,0
WHITE = 255,255,255
RED = 0,0,255
ACCENT = 0,255,0
BLUE = 255,0,0
GRAY = 100,100,100
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIZE = 0.4
VISUALIZE = True
device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Craftax-Symbolic-v1")
parser.add_argument("--model", type=str, default="models/best.sav")
parser.add_argument("--video", type=str, default="videos/last.mp4")
parser.add_argument("--fps", type=int, default=3)
parser.add_argument("--video_fps", type=int, default=6)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--world", type=int, default=-1)
parser.add_argument("--attention", action="store_true")
parser.add_argument("--gamma", type=float, default=0.3)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--argmax", action="store_true")
parser.add_argument("--frame", action="store_true")

args, rest_args = parser.parse_known_args(sys.argv[1:])
if rest_args:
    raise ValueError(f"Unknown args {rest_args}")

agent = load_agent_from_save(args.model, parallel=1, device=device)
agent.eval()
load_time = 0

seed = args.seed if args.seed >= 0 else np.random.randint(2**31)
torch.set_printoptions(precision=3)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

env = CraftaxSymbolic()
obs = env.reset(seed)

renderer = Renderer(upscale=args.scale)
game = renderer.render(env.state) # (width=704, height=832, 3)
info = np.zeros((400, game.shape[1], 3), dtype=np.uint8)
frame = np.concatenate((game, info), axis=0)
video_shape = game.shape[0], game.shape[1]
if args.frame:
    video_shape = frame.shape[0], frame.shape[1]

info = info.transpose(1, 0, 2)

codec = cv2.VideoWriter_fourcc(*'mp4v')

hi_score = 0

world = -args.world - 1 if args.world < 0 else args.world

episode = 1

while True:

    try:
        mtime = os.path.getmtime(args.model) 
        if mtime > load_time:
            try:
                save = torch.load(args.model, map_location=device, weights_only=False)
                load_time = mtime
            except Exception as error:
                raise error
    except Exception as error:
        raise error

    video = cv2.VideoWriter(args.video, codec, args.video_fps, video_shape)

    agent.load_state_dict(save['agent_state'])

    if args.world < 0:
        world += 1

    obs = env.reset(world)

    if args.seed < 0:
        seed = np.random.randint(2**31)

    if world >= 0:
        torch.manual_seed(seed)

    reward_sum = 0
    done = 0
    step = 0

    while not done:
        done = np.array([done], dtype=bool)
        obs = torch.Tensor(np.array(obs.reshape(1, -1)))
        belief = agent.belief()

        with torch.no_grad():
            state, attention = agent.state(obs, belief, return_attention=True)
            policy = agent.mask(agent.actor(state), obs).softmax(-1).reshape(-1)
            if args.argmax:
                action = int(torch.argmax(policy))
            else:
                action = int(torch.multinomial(policy, 1)[0])
            critic = float(agent.critic(state)[0,0])

        achv_old = env.state.achievements
        old_state = env.state
        next_obs, reward, done, step_info = env.step(action)
        #rewards = torch.Tensor([reward])
        #print(done)
        #terminal = torch.Tensor([done]).bool()
        #agent.update(obs, action, rewards, terminal, terminal, step_info, next_obs)
        achv_new = env.state.achievements
        step += 1
        reward_sum += reward
        score = 100*reward_sum/226
        obs = next_obs

        action = Action(action)

        if not VISUALIZE:
            continue

        #print(f'EP {episode} DEPTH {depth} REWARD {reward_sum:.2f} CRITIC {critic:.2f} - {action.name} ({reward:+.2f}) ')
        game = renderer.render(old_state)
        game = game.transpose((1, 0, 2))
        game = cv2.cvtColor(game, cv2.COLOR_RGB2BGR)

        if attention is not None:
            repeat = 16*args.scale
            attn = np.repeat(np.repeat(attention.view(9, 11).numpy(), repeat, axis=0), repeat, axis=1)
            mask = 0*game + 255
            alpha = np.zeros((game.shape[0], game.shape[1]))
            alpha[:9*repeat] = attn
            alpha = alpha[...,None]
            game = (1 - alpha)*game + alpha*mask
            game = game.astype(np.uint8)

        if args.frame:
            info = info.copy() * 0
            y = 20
            cv2.putText(info, f'ACTIONS', (10, y), FONT, SIZE, WHITE)
            y += 10
            for prob, x in zip(policy, Action):
                fg = WHITE if x == action else GRAY
                bg = (0,100,0)
                hist = int(math.ceil(200*prob))
                cv2.rectangle(info, (10,y-10), (10+hist,y), bg, -1)
                cv2.putText(info, x.name, (10, y), FONT, SIZE, fg)
                y += 10
            y += 10; cv2.putText(info, f'SEED {seed}', (10, y), FONT, SIZE, WHITE)
            y += 20; cv2.putText(info, f'WORLD {world}', (10, y), FONT, SIZE, WHITE)
            y += 20; cv2.putText(info, f'STEP {step}', (10, y), FONT, SIZE, WHITE)
            y += 20; cv2.putText(info, f'CRITIC {critic:.2f}', (10, y), FONT, SIZE, WHITE)
            y += 20; cv2.putText(info, f'REWARD {reward:+.2f}', (10, y), FONT, SIZE, WHITE)
            y += 20; cv2.putText(info, f'SCORE {score:.2f}% ({reward_sum:.2f}/226)', (10, y), FONT, SIZE, ACCENT)
            y = 20
            achv_count = int(achv_new.sum())
            cv2.putText(info, f'ACHIEVEMENTS {achv_count}/{len(Achievement)}', (200+10, y), FONT, SIZE, WHITE)
            y += 10
            for i in range(len(achv_new)):
                x = Achievement(i)
                if achv_new[i] == 1:
                    cv2.putText(info, 'DONE', (400-40, y), FONT, SIZE, ACCENT)
                cv2.putText(info, x.name, (200+10, y), FONT, SIZE, GRAY)
                y += 10
            frame = np.concatenate((game, info), axis=1)
        else:
            frame = game

        video.write(frame)
        cv2.imshow(args.model, frame)
        sleeping = obs[8261] > 0
        fps = 60 if sleeping else args.fps
        wait_time = 3000 if done else 1000 // fps
        cv2.waitKey(wait_time)

    video.release()

    print(f'EP {episode} WORLD {world} SCORE {score:.2f}%')
    if score > hi_score:
        hi_score = score
        new_path = os.path.join(os.path.dirname(args.video), f'{int(score)}_w{world}s{seed}.mp4')
        os.rename(args.video, new_path)
        print('SAVED', new_path)
    elif achv_new[Achievement.ENTER_GNOMISH_MINES.value] == 1:
        new_path = os.path.join(os.path.dirname(args.video), f'{int(score)}-mines_w{world}s{seed}.mp4')
        os.rename(args.video, new_path)
        print('SAVED', new_path)

    episode += 1

print(f'{hi_score:.2f}')
cv2.destroyAllWindows()
