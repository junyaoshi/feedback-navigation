# Import Libraries
from gibson.envs.husky_env import Husky2DNavigateEnv
from gibson.utils.play import play
import inspect
from pprint import pprint
import pybullet as p
import os
import time
import math
import random
import time as t
from time import sleep
from math import pi, degrees

# Parse config and object arguments
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7_ppo2_2D.yaml')
pprint('config file: {}'.format(config_file))

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default=config_file)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print(args)


env = Husky2DNavigateEnv(config=args.config, gpu_idx=args.gpu)
env.reset()

# env.reset_goal_range(goal_range=1.5)
# env.reset_state_space(raycast_num=1)
# env._show_rays = True
#
for i in range(10000):
    start = time.time()
    state = env.get_robot_state()
    # for action in range(env.action_space.n):
    #     # env.step(action)
    #     if env.judge_action(state, action):
    #        good_actions.append(action)
    good_actions = env.get_good_actions(state)
    bad_actions = []
    for a in range(env.action_space.n):
        if a not in good_actions:
            bad_actions.append(a)

    # print('good actions: {}'.format(good_actions))
    rand_num = random.random()
    if rand_num > 0.5 and good_actions:
        obs, rew, env_done, _ = env.step(random.choice(good_actions))
    elif rand_num <= 0.5 and bad_actions:
        obs, rew, env_done, _ = env.step(random.choice(bad_actions))
    else:
        obs, rew, env_done, _ = env.step(random.choice(range(env.action_space.n)))

    print('obs_len: {}, obs_space_len: {}, obs: {}'.format(len(obs['obs']), env.observation_space.shape, obs['obs']))

    end = time.time()
    print('this step took {} seconds'.format(end-start))
#
#     if env_done:
#         env.reset()
# #     num = random.random()
# #     # env.step(3)
# #     if num < 0.25:
# #         env.step(0)
# #     elif num < 0.5:
# #         env.step(1)
# #     elif num < 0.75:
# #         env.step(2)
# #     else:
# #         env.step(3)
# #     for i in range(5):
# #         env.step(4)
#
#     # pressed_keys = env.get_key_pressed()
#     # for key in pressed_keys:
#     #     processing = True
#     #     if not processing:
#     #         break
#     #     if key == ord(' '):
#     #         pausing = True
#     #         while pausing:
#     #             time.sleep(0.1)
#     #             pressed_keys = env.get_key_pressed()
#     #             for key in pressed_keys:
#     #                 if key == ord(' '):
#     #                     pressed_keys = []
#     #                     pausing = False
#     #                     processing = False
#     #                     break
#
#
#
#
# # for i in range(10000):
# #     env.step(3)
# #     time.sleep(2)




play(env)

import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat
from gibson.core.physics.robot_bases import quatToXYZW
env.reset()
x_init, y_init = -10, 18.8
env.robot.robot_body.reset_position([x_init, y_init, 0.14])
env.robot.robot_body.reset_orientation(quatToXYZW(euler2quat(*[0, 0, np.pi/3]), 'wxyz'))
#
if hasattr(env, 'get_keys_to_action'):
    keys_to_action = env.get_keys_to_action()
elif hasattr(env.unwrapped, 'get_keys_to_action'):
    keys_to_action = env.unwrapped.get_keys_to_action()
action = keys_to_action[()]
obs, rew, env_done, info = env.step(action)

start = t.time()
# i = 0
# while True:
for i in range(2000):
    print(env.robot.robot_body.speed(), env.robot.get_position())
    env.step(env.IDLE)
    # print(obs)
    # print('robot xyz {}  rpy {}'.format(env.robot.get_position(), env.robot.get_rpy()))
    x, y, z = env.robot.get_position()
    if (x - x_init) ** 2 + (y - y_init) ** 2 + (z - 0.14) ** 2 > 1 * 0.05 ** 2: #
        print(i, t.time() - start)
        # break
    # aabbMin, aabbMax = p.getAABB(env.robot_tracking_id)
    # print(len(p.getOverlappingObjects(aabbMin, aabbMax)), p.getOverlappingObjects(aabbMin, aabbMax))

