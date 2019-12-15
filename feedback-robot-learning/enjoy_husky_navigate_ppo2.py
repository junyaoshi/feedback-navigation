# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import gym
import logging
import random
from mpi4py import MPI
from baselines.common import set_global_seeds
from gibson.utils import utils
from baselines import logger
from gibson.utils.monitor import Monitor

# TODO: merge imports
from gibson.envs.husky_env import Husky1DNavigateEnv, Husky2DNavigateEnv
from train_ppo.ppo_trainer import enjoy
from train_ppo.feedback_policy import FeedbackPolicy
import json
import matplotlib.pyplot as plt

SEED = random.randint(0, 20000)
RELOAD_DIR = os.path.join('simulation_and_analysis', 'rslts', 'husky_ppo2_hr_D190818_044225')
TOTAL_TIMESTEPS = 3000


def enjoy_husky():
    rank = MPI.COMM_WORLD.Get_rank()
    sess = utils.make_gpu_session(args.num_gpu)
    sess.__enter__()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    
    param_fname = os.path.join(args.reload_dir, 'param.json')
    with open(param_fname, 'r') as f:
        param = json.load(f)

    performance_fname = os.path.join(args.reload_dir, 'performance.p')

    if param['use_2D_env']:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7_ppo2_2D.yaml')
        raw_env = Husky2DNavigateEnv(gpu_idx=args.gpu_idx,
                                     config=config_file,
                                     pos_interval=param['pos_interval'])
    else:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7_ppo2.yaml')
        raw_env = Husky1DNavigateEnv(gpu_idx=args.gpu_idx,
                                     config=config_file,
                                     ob_space_range=[0.0, 40.0])

    # configure environment
    raw_env.reset_state_space(use_goal_info=param["use_goal_info"],
                              use_coords_and_orn=param["use_coords_and_orn"],
                              raycast_num=param["raycast_num"],
                              raycast_range=param["raycast_range"])
    raw_env.reset_goal_range(goal_range=param["goal_range"])

    env = Monitor(raw_env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    gym.logger.setLevel(logging.WARN)

    policy_fn = FeedbackPolicy

    enjoy(policy=policy_fn, env=env, total_timesteps=args.total_timesteps, base_path=args.reload_dir,
          ent_coef=param["ent_coef"], vf_coef=0.5, max_grad_norm=param['max_grad_norm'], gamma=param["gamma"], lam=param["lambda"],
          nsteps=param['nsteps'],
          ppo_minibatch_size=param['ppo_minibatch_size'], feedback_minibatch_size=param['feedback_minibatch_size'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--reload_dir', type=str, default=RELOAD_DIR)
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    
    assert args.reload_dir is not None

    enjoy_husky()
