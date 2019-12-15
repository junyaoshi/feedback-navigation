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
import json
import pickle

from train_dqn.deepq import learn
import pybullet as p


SEED = random.randint(0, 20000)

# ------------------------------------------- env ------------------------------------------- #
USE_2D_ENV = True                   # whether use 1D or 2D Gibson environment
USE_RICH_REWARD = False             # whether the environment uses rich reward
USE_MULTIPLE_STARTS = False         # whether the environment has multiple starting positions
POS_INTERVAL = 0.3                  # how far does the car go when moving forward

# ----------------------------------------- feedback ---------------------------------------- #
USE_FEEDBACK = True                 # whether use feedback to learn a HR policy
USE_REAL_FEEDBACK = False           # whether use feedback based on BCI
TRANS_BY_INTERPOLATE = False        # whether transition from HR to RL in policy shaping style
GOOD_FEEDBACK_ACC = 0.7             # simulated feedback accuracy for good actions
BAD_FEEDBACK_ACC = 0.7              # simulated feedback accuracy for bad actions

# ----------------------------------------- step num ---------------------------------------- #
ONLY_USE_HR_UNTIL = 1000            # only use HR until xxx steps
TRANS_TO_RL_IN = int(2e4)           # transition from HR to RL in xxx steps
TOTAL_TIEMSTEPS = int(4e4)          # total num of steps the experiment runs

# ------------------------------------------- DQN ------------------------------------------- #
EXPLORATION_FRACTION = 0.2
EXPLORATION_FINAL_EPS = 0.1
LR = 5e-4
BATCH_SIZE = 32
DQN_EPOCHS = 16
TRAIN_FREQ = 1
TARGET_NETWORK_UPDATE_FREQ = 32
LEARNING_STARTS = 1000
PARAM_NOISE = True
GAMMA = 1.0

#  ------------------------------------ DQN replay buffer ----------------------------------- #
BUFFER_SIZE = 50000
PRIORITIZED_REPLAY = True
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA0 = 0.4
PRIORITIZED_REPLAY_BETA_ITERS = None
PRIORITIZED_REPLAY_EPS = 1e-6

# ------------------------------ feedback function approximator ----------------------------- #
FEEDBACK_LR = 1e-3                  # feedback function approximator (FFA) learning rate
FEEDBACK_EPOCHS = 20                # num of epochs during one FFA update
FEEDBACK_BATCH_SIZE = 8             # num of steps to collect for one FFA update
FEEDBACK_MINIBATCH_SIZE = 4         # num of steps in a minibatch
FEEDBACK_TRAINING_PROP = 0.8        # the proportion of feedbacks were used as training data
FEEDBACK_TRAINING_NEW_PROP = 0.4    # the proportion of epochs were trained on newly received feedbacks
MIN_FEEDBACK_BUFFER_SIZE = 40       # min size of feedback buffer

CHECKPOINT_FREQ = 10

def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    import datetime
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date


def train():
    rank = MPI.COMM_WORLD.Get_rank()
    sess = utils.make_gpu_session(args.num_gpu)
    sess.__enter__()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    if args.use_2D_env:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7_ppo2_2D.yaml')
    else:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7_ppo2.yaml')

    if args.use_2D_env:
        raw_env = Husky2DNavigateEnv(gpu_idx=args.gpu_idx,
                                     config=config_file,
                                     pos_interval=args.pos_interval)
    else:
        raw_env = Husky1DNavigateEnv(gpu_idx=args.gpu_idx,
                                     config=config_file,
                                     ob_space_range=[0.0, 40.0])

    env = Monitor(raw_env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    gym.logger.setLevel(logging.WARN)

    base_dirname = os.path.join(currentdir, "simulation_and_analysis_dqn", "rslts")

    if not os.path.exists(base_dirname):
        os.makedirs(base_dirname)
    dir_name = "husky_dqn_"
    if args.use_feedback:
        dir_name += "hr"
    elif args.use_rich_reward:
        dir_name += "rl_rich"
    else:
        dir_name += "rl_sparse"
    dir_name = addDateTime(dir_name)
    dir_name = os.path.join(base_dirname, dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


    hyperparams = {
        "seed": args.seed,
        # env
        "use_2D_env": args.use_2D_env,
        "use_rich_reward": args.use_rich_reward,
        "use_multiple_starts": args.use_multiple_starts,
        "total_timesteps": args.total_timesteps,
        "pos_interval": args.pos_interval,
        # hr
        "use_feedback": args.use_feedback,
        "use_real_feedback": args.use_real_feedback,
        "trans_by_interpolate": args.trans_by_interpolate,
        "only_use_hr_until": args.only_use_hr_until,
        "trans_to_rl_in": args.trans_to_rl_in,
        "good_feedback_acc": args.good_feedback_acc,
        "bad_feedback_acc": args.bad_feedback_acc,
        # dqn
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dqn_epochs": args.dqn_epochs,
        "train_freq": args.train_freq,
        "target_network_update_freq": args.target_network_update_freq,
        "learning_starts": args.learning_starts,
        "param_noise": args.param_noise,
        "gamma": args.gamma,
        # hr training
        "feedback_lr": args.feedback_lr,
        "feedback_epochs": args.feedback_epochs,
        "feedback_batch_size": args.feedback_batch_size,
        "feedback_minibatch_size": args.feedback_minibatch_size,
        "min_feedback_buffer_size": args.min_feedback_buffer_size,
        "feedback_training_prop": args.feedback_training_prop,
        "feedback_training_new_prop": args.feedback_training_new_prop,
        # dqn replay buffer
        "buffer_size": args.buffer_size,
        "prioritized_replay": args.prioritized_replay,
        "prioritized_replay_alpha": args.prioritized_replay_alpha,
        "prioritized_replay_beta0": args.prioritized_replay_beta0,
        "prioritized_replay_beta_iters": args.prioritized_replay_beta_iters,
        "prioritized_replay_eps": args.prioritized_replay_eps,
        #
        "checkpoint_freq": args.checkpoint_freq,
        "use_embedding": raw_env._use_embedding,
        "use_raycast": raw_env._use_raycast,
        "offline": raw_env.config['offline']
    }

    print_freq = 5

    param_fname = os.path.join(dir_name, "param.json")
    with open(param_fname, "w") as f:
        json.dump(hyperparams, f, indent=4, sort_keys=True)

    video_name = os.path.join(dir_name, "video.mp4")
    p_logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_name)

    act, performance = learn(# env flags
                              env,
                              raw_env,
                              use_2D_env=args.use_2D_env,
                              use_multiple_starts=args.use_multiple_starts,
                              use_rich_reward=args.use_rich_reward,
                              total_timesteps=args.total_timesteps,
                              # dqn
                              exploration_fraction=args.exploration_fraction,
                              exploration_final_eps=args.exploration_final_eps,
                              # hr
                              use_feedback=args.use_feedback,
                              use_real_feedback=args.use_real_feedback,
                              only_use_hr_until=args.only_use_hr_until,
                              trans_to_rl_in=args.trans_to_rl_in,
                              good_feedback_acc=args.good_feedback_acc,
                              bad_feedback_acc=args.bad_feedback_acc,
                              # dqn training
                              lr=args.lr,
                              batch_size=args.batch_size,
                              dqn_epochs=args.dqn_epochs,
                              train_freq=args.train_freq,
                              target_network_update_freq=args.target_network_update_freq,
                              learning_starts=args.learning_starts,
                              param_noise=args.param_noise,
                              gamma=args.gamma,
                              # hr training
                              feedback_lr=args.feedback_lr,
                              feedback_epochs=args.feedback_epochs,
                              feedback_batch_size=args.feedback_batch_size,
                              feedback_minibatch_size=args.feedback_minibatch_size,
                              min_feedback_buffer_size=args.min_feedback_buffer_size,
                              feedback_training_prop=args.feedback_training_prop,
                              feedback_training_new_prop=args.feedback_training_new_prop,
                              # replay buffer
                              buffer_size=args.buffer_size,
                              prioritized_replay=args.prioritized_replay,
                              prioritized_replay_alpha=args.prioritized_replay_alpha,
                              prioritized_replay_beta0=args.prioritized_replay_beta0,
                              prioritized_replay_beta_iters=args.prioritized_replay_beta_iters,
                              prioritized_replay_eps=args.prioritized_replay_eps,
                              # rslts saving and others
                              checkpoint_freq=args.checkpoint_freq,
                              print_freq=print_freq,
                              checkpoint_path=None,
                              load_path=None,
                              callback=None,
                              seed=args.seed)

    p.stopStateLogging(p_logging)

    performance_fname = os.path.join(dir_name, "performance.p")
    with open(performance_fname, "wb") as f:
        pickle.dump(performance, f)
    act.save(os.path.join(dir_name, "cartpole_model.pkl"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--reload_name', type=str, default=None)

    parser.add_argument('--use_2D_env', action='store_true', default=USE_2D_ENV)
    parser.add_argument('--use_rich_reward', action='store_true', default=USE_RICH_REWARD)
    parser.add_argument('--use_multiple_starts', action='store_true', default=USE_MULTIPLE_STARTS)
    parser.add_argument('--pos_interval', type=float, default=POS_INTERVAL)
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIEMSTEPS)

    parser.add_argument('--use_feedback', action='store_true', default=USE_FEEDBACK)
    parser.add_argument('--use_real_feedback', action='store_true', default=USE_REAL_FEEDBACK)
    parser.add_argument('--only_use_hr_until', type=int, default=ONLY_USE_HR_UNTIL)
    parser.add_argument('--trans_to_rl_in', type=int, default=TRANS_TO_RL_IN)
    parser.add_argument('--good_feedback_acc', type=float, default=GOOD_FEEDBACK_ACC)
    parser.add_argument('--bad_feedback_acc', type=float, default=BAD_FEEDBACK_ACC)
    parser.add_argument('--trans_by_interpolate', action='store_true', default=TRANS_BY_INTERPOLATE)

    parser.add_argument('--exploration_fraction', type=float, default=EXPLORATION_FRACTION)
    parser.add_argument('--exploration_final_eps', type=float, default=EXPLORATION_FINAL_EPS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--dqn_epochs', type=int, default=DQN_EPOCHS)
    parser.add_argument('--train_freq', type=int, default=TRAIN_FREQ)
    parser.add_argument('--target_network_update_freq', type=int, default=TARGET_NETWORK_UPDATE_FREQ)
    parser.add_argument('--learning_starts', type=int, default=LEARNING_STARTS)
    parser.add_argument('--param_noise', action='store_true', default=PARAM_NOISE)
    parser.add_argument('--gamma', type=float, default=GAMMA)

    parser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE)
    parser.add_argument('--prioritized_replay', action='store_true', default=PRIORITIZED_REPLAY)
    parser.add_argument('--prioritized_replay_alpha', type=float, default=PRIORITIZED_REPLAY_ALPHA)
    parser.add_argument('--prioritized_replay_beta0', type=float, default=PRIORITIZED_REPLAY_BETA0)
    parser.add_argument('--prioritized_replay_beta_iters', type=float, default=PRIORITIZED_REPLAY_BETA_ITERS)
    parser.add_argument('--prioritized_replay_eps', type=float, default=PRIORITIZED_REPLAY_EPS)

    parser.add_argument('--feedback_lr', type=float, default=FEEDBACK_LR)
    parser.add_argument('--feedback_epochs', type=int, default=FEEDBACK_EPOCHS)
    parser.add_argument('--feedback_batch_size', type=int, default=FEEDBACK_BATCH_SIZE)
    parser.add_argument('--feedback_minibatch_size', type=int, default=FEEDBACK_MINIBATCH_SIZE)
    parser.add_argument('--min_feedback_buffer_size', type=int, default=MIN_FEEDBACK_BUFFER_SIZE)
    parser.add_argument('--feedback_training_prop', type=float, default=FEEDBACK_TRAINING_PROP)
    parser.add_argument('--feedback_training_new_prop', type=float, default=FEEDBACK_TRAINING_NEW_PROP)

    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ)

    parser.add_argument('--seed', type=int, default=SEED)

    args = parser.parse_args()

    train()

