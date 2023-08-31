import numpy as np

from catch_ur3.envs.catch_ur3 import CatchUR3Env


def test(args):

    render_mode = args.render_mode
    num_episode = args.num_test_episode

    env = CatchUR3Env(render_mode=render_mode)

    for idx_episode in range(num_episode):

        obs, info = env.reset()
        qpos = info['ur3_qpos']
        increment = np.ones(env.ur3_nqpos) * -0.01

        for i in range(100):
            action = env.servoj(qpos+increment, 0, 0)
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            qpos = info['ur3_qpos']

            if render_mode:
                frame = env.render()
    
    env.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test_episode', type=int, default=10, help='N of test episodes to run (default: 10)')
    parser.add_argument('--render_mode', type=str, choices=['human', 'rgb_array', 'depth'], default='human', help='Rendering mode (default: human)')

    args = parser.parse_args()

    test(args)