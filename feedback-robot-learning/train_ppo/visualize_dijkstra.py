import pickle
import numpy as np
import matplotlib.pyplot as plt
from gibson.envs.husky_env import Husky2DNavigateEnv
import train_ppo
import os

with open("husky_navigate_2D_dijkstra.p", "rb") as f:
    dij = pickle.load(f)

dist = dij["dist"]
collision = dij["collision"].astype(int)

# Parse config and object arguments
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'husky_space7_ppo2_2D.yaml')

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default=config_file)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print(args)

env = Husky2DNavigateEnv(config=args.config, gpu_idx=args.gpu)
env.reset()

x_range, y_range, yaw_range = dist.shape
print(x_range, y_range, yaw_range)

for yaw_idx in range(yaw_range):
    plt.imshow(collision[:, :, yaw_idx])

    xs = []
    ys = []
    for x_idx in range(x_range):
        for y_idx in range(y_range):
            print('at {}, {}, {}'.format(x_idx, y_idx, yaw_idx))
            x, y, yaw = env.idx_2_cor(x_idx, y_idx, yaw_idx)
            good_actions = env.get_good_actions([x, y, yaw])
            if not collision[x_idx, y_idx, yaw_idx] and not good_actions:
                xs.append(x_idx)
                ys.append(y_idx)
    plt.scatter(xs, ys, c='r', s=0.2)
    # plt.colorbar(orientation='vertical')

plt.show()
plt.savefig("problematic points in dijkstra")


# dist = dij["dist"]
# dist = np.clip(dist, 0, 40)
# for i in range(12):
#     print(i)
#     plt.imshow(dist[:, :, i])
#     plt.colorbar(orientation='vertical')
#     plt.show()

# collision = (1 - dij["collision"].astype(int)).astype(bool)
# dij["collision"] = collision
# with open("husky_navigate_2D_dijkstra.p", "wb") as f:
#     pickle.dump(dij, f)
