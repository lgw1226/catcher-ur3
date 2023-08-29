from copy import copy
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from catch_ur3.envs.catch_ur3 import CatchUR3Env


class PosSequence:
    def __init__(self, maxlen=5):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def push(self, pos):
        self.buffer.append(pos)

    def get_data(self):
        return np.array(self.buffer)


def tidy_center_print(val='', terminal_width=80):

    print(f"{val:-^{terminal_width}}")

def tidy_key_val_print(key, val, terminal_width=80):
    
    leftover_width = terminal_width - len(key)

    print(f"{key}{val:>{leftover_width}.4f}")

def estimate_trajectory(
    pos_sequence,
    p1=None,  # parameters for straignt line on xy-plane
    p2=None,  # parameters for parabola on sz-plane
    max_iter=100,
    alpha=0.005,
    eps=0.001
):
    
    pos_sequence = torch.tensor(pos_sequence)
    len_sequence = len(pos_sequence)

    # fit a straight line on xy-plane
    xy1 = torch.cat((pos_sequence[:,0:2], torch.ones(len_sequence, 1)), axis=1)

    if not p1:  # if there's no initial guess
        p1 = torch.randn(3, dtype=torch.float64, requires_grad=True)  # a_1 * x + a_2 * y + a_3 = 0
    else:  # typecasting for input check
        p1 = torch.tensor(p1, requires_grad=True)

    loss_old = torch.zeros(1)

    for _ in range(max_iter):

        loss = torch.mean(torch.abs(xy1 @ p1) / torch.sqrt(p1[0] ** 2 + p1[1] ** 2))
        print(loss)
        if torch.abs(loss - loss_old) <= eps:
            print('broken')
            break
        
        loss.backward()
        loss_old = loss

        with torch.no_grad():
            p1_new = p1 - alpha * p1.grad
            p1 = p1_new.requires_grad_()

    # projection of xy-points on the straight line
    

    # fit a parabola on sz-plane


env = CatchUR3Env()

num_episode = 1
len_ball_pos_sequence = 5  # number of coordinates that make up a sequence

tidy_center_print('Info')
tidy_key_val_print("Time elapsed between two consecutive simulation frames (sec)", env.dt)
tidy_key_val_print("Number of coordinates used in computing parabolic trajectory (#)", len_ball_pos_sequence)
tidy_center_print()

for idx_episode in range(num_episode):

    obs, info = env.reset()

    ball_pos_sequence = PosSequence()
    ball_pos_sequence.push(copy(info['ball_pos']))

    for idx_step in range(10):
        action = np.zeros(6)
        obs, reward, terminated, truncated, info = env.step(action)
        ball_pos_sequence.push(copy(info['ball_pos']))

        if idx_step >= len_ball_pos_sequence:
            estimate_trajectory(ball_pos_sequence.get_data())
 
env.close()

