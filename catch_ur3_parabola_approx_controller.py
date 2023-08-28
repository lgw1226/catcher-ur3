from copy import copy
from collections import deque

import numpy as np

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

def estimate_trajectory(pos_sequence):

    x = pos_sequence[:,0]
    y = pos_sequence[:,1]
    z = pos_sequence[:,2]

    


env = CatchUR3Env()

num_episode = 5
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
            print(ball_pos_sequence.get_data())
 
env.close()

