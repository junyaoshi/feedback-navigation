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
import pybullet as p

# TODO: merge imports
from gibson.envs.husky_env import Husky1DNavigateEnv, Husky2DNavigateEnv
from train_ppo.ppo_trainer import learn
from train_ppo.feedback_policy import FeedbackPolicy
import json
import pickle

SEED = random.randint(0, 20000)

# ------------------------------------------- env ------------------------------------------- #
USE_2D_ENV = True                   # whether use 1D or 2D Gibson environment
USE_OTHER_ROOM = False              # whether to use the other room
USE_RICH_REWARD = False             # whether the environment uses rich reward
USE_MULTIPLE_STARTS = False         # whether the environment has multiple starting positions
POS_INTERVAL = 0.3                  # how far does the car go when moving forward
USE_GOAL_INFO = True               # whether use husky's polar coordinates from goal in state space
USE_COORDS_AND_ORN = False          # whether use husky's x, y, theta in state space
RAYCAST_NUM = 10                    # number of ray casted for ray info in state space
RAYCAST_RANGE = 6                   # range of ray cast towards front of car, unit is pi/6 radian
GOAL_RANGE = 0.5                    # tolerance range for determining whether or not husky has reached goal
START_VARYING_RANGE = 0.2

# ----------------------------------------- feedback ---------------------------------------- #
USE_FEEDBACK = False               # whether use feedback to learn a HR policy
USE_REAL_FEEDBACK = False           # whether use feedback based on BCI
GOOD_FEEDBACK_ACC = 0.7             # simulated feedback accuracy for good actions
BAD_FEEDBACK_ACC = 0.7              # simulated feedback accuracy for bad actions

# ----------------------------------------- step num ---------------------------------------- #
NSTEPS = 2                          # num of steps each time runner runs
ONLY_USE_HR_UNTIL = 1000            # only use HR until xxx steps
TRANS_TO_RL_IN = int(2e4)           # transition from HR to RL in xxx steps
TOTAL_TIEMSTEPS = int(4e4)          # total num of steps the experiment runs

# ------------------------------------------- PPO ------------------------------------------- #
PPO_LR = 3e-4                       # PPO learning rate
ENT_COEF = 0.02                     # entropy coefficient in PPO loss
GAMMA = 0.99
LAMBDA = 0.95
CLIPRANGE = 0.2
MAX_GRAD_NORM = 25.0
PPO_NOPTEPOCHS = 64                 # num of epochs during one PPO update
PPO_BATCH_SIZE = 128                # num of steps to collect for one PPO update
PPO_MINIBATCH_SIZE = 64             # num of steps in a minibatch
INIT_RL_IMPORTANCE = 0.2            # the initial on-policy rl importance, will linearly increase to 1 during trans_to_rl

# ------------------------------ feedback function approximator ----------------------------- #
FEEDBACK_LR = 1e-4                  # feedback function approximator (FFA) learning rate
FEEDBACK_NOPTEPOCHS = 16            # num of epochs during one FFA update
FEEDBACK_BATCH_SIZE = 64             # num of steps to collect for one FFA update
FEEDBACK_MINIBATCH_SIZE = 32         # num of steps in a minibatch
FEEDBACK_TRAINING_PROP = 0.8        # the proportion of feedbacks were used as training data
FEEDBACK_TRAINING_NEW_PROP = 0.3    # the proportion of epochs were trained on newly received feedbacks
MIN_FEEDBACK_BUFFER_SIZE = 64       # min size of feedback buffer
HF_LOSS_TYPE = "CCE"
HF_LOSS_PARAM = "0.2,0.3"
FEEDBACK_USE_MIXUP = False



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
                                     pos_interval=args.pos_interval,
                                     use_other_room=args.use_other_room)
    else:
        raw_env = Husky1DNavigateEnv(gpu_idx=args.gpu_idx,
                                     config=config_file,
                                     ob_space_range=[0.0, 40.0])

    # configure environment
    raw_env.reset_state_space(use_goal_info=args.use_goal_info,
                              use_coords_and_orn=args.use_coords_and_orn,
                              raycast_num=args.raycast_num,
                              raycast_range=args.raycast_range,
                              start_varying_range=args.start_varying_range)
    raw_env.reset_goal_range(goal_range=args.goal_range)

    env = Monitor(raw_env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    gym.logger.setLevel(logging.WARN)

    policy_fn = FeedbackPolicy

    base_dirname = os.path.join(currentdir, "simulation_and_analysis", "rslts")

    if not os.path.exists(base_dirname):
        os.mkdir(base_dirname)
    dir_name = "husky_ppo2_"
    if args.use_feedback:
        dir_name += "hr"
        if args.use_real_feedback:
            dir_name += "_real_feedback"

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
        "nsteps": args.nsteps,
        "total_timesteps": args.total_timesteps,
        "use_2D_env": args.use_2D_env,
        "use_other_room": args.use_other_room,
        "use_rich_reward": args.use_rich_reward,
        "use_multiple_starts": args.use_multiple_starts,
        "use_goal_info": args.use_goal_info,
        "use_coords_and_orn": args.use_coords_and_orn,
        "raycast_num": args.raycast_num,
        "raycast_range": args.raycast_range,
        "goal_range": args.goal_range,
        "start_varying_range": args.start_varying_range,
        "use_feedback": args.use_feedback,
        "use_real_feedback": args.use_real_feedback,
        "only_use_hr_until": args.only_use_hr_until,
        "trans_to_rl_in": args.trans_to_rl_in,
        "good_feedback_acc": args.good_feedback_acc,
        "bad_feedback_acc": args.bad_feedback_acc,
        "ppo_lr": args.ppo_lr,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        "init_rl_importance": args.init_rl_importance,
        "ent_coef": args.ent_coef,
        "gamma": args.gamma,
        "lambda": args.lam,
        "cliprange": args.cliprange,
        "max_grad_norm": args.max_grad_norm,
        "ppo_noptepochs": args.ppo_noptepochs,
        "feedback_lr": args.feedback_lr,
        "feedback_batch_size": args.feedback_batch_size,
        "feedback_minibatch_size": args.feedback_minibatch_size,
        "feedback_noptepochs": args.feedback_noptepochs,
        "min_feedback_buffer_size": args.min_feedback_buffer_size,
        "feedback_training_prop": args.feedback_training_prop,
        "feedback_training_new_prop": args.feedback_training_new_prop,
        "hf_loss_type": args.hf_loss_type,
        "hf_loss_param": args.hf_loss_param,
        "feedback_use_mixup": args.feedback_use_mixup,
        "pos_interval": args.pos_interval,
        "use_embedding": raw_env._use_embedding,
        "use_raycast": raw_env._use_raycast,
        "offline": raw_env.config['offline']
    }

    param_fname = os.path.join(dir_name, "param.json")
    with open(param_fname, "w") as f:
        json.dump(hyperparams, f, indent=4, sort_keys=True)

    hf_loss_param = [float(x) for x in args.hf_loss_param.split(",") if x != ""]

    video_name = os.path.join(dir_name, "video.mp4")
    p_logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_name)

    performance = learn(policy=policy_fn, env=env, raw_env=raw_env,
                        use_2D_env=args.use_2D_env,
                        use_other_room=args.use_other_room,
                        use_multiple_starts=args.use_multiple_starts,
                        use_rich_reward=args.use_rich_reward,
                        use_feedback=args.use_feedback,
                        use_real_feedback=args.use_real_feedback,
                        only_use_hr_until=args.only_use_hr_until,
                        trans_to_rl_in=args.trans_to_rl_in,
                        nsteps=args.nsteps,
                        total_timesteps=args.total_timesteps,
                        ppo_lr=args.ppo_lr, cliprange=args.cliprange, max_grad_norm=args.max_grad_norm,
                        ent_coef=args.ent_coef, gamma=args.gamma, lam=args.lam,
                        ppo_noptepochs=args.ppo_noptepochs,
                        ppo_batch_size=args.ppo_batch_size, ppo_minibatch_size=args.ppo_minibatch_size,
                        init_rl_importance=args.init_rl_importance,
                        feedback_lr=args.feedback_lr,
                        feedback_noptepochs=args.feedback_noptepochs,
                        feedback_batch_size=args.feedback_batch_size, feedback_minibatch_size=args.feedback_minibatch_size,
                        min_feedback_buffer_size=args.min_feedback_buffer_size,
                        feedback_training_prop=args.feedback_training_prop,
                        feedback_training_new_prop=args.feedback_training_new_prop,
                        hf_loss_type=args.hf_loss_type, hf_loss_param=hf_loss_param,
                        feedback_use_mixup=args.feedback_use_mixup,
                        good_feedback_acc=args.good_feedback_acc,
                        bad_feedback_acc=args.bad_feedback_acc,
                        log_interval=1, save_interval=5, reload_name=None, base_path=dir_name)

    p.stopStateLogging(p_logging)

    performance_fname = os.path.join(dir_name, "performance.p")
    with open(performance_fname, "wb") as f:
        pickle.dump(performance, f)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(performance["train_acc"], label="train_acc")
    plt.plot(performance["train_true_acc"], label="train_true_acc")
    plt.plot(performance["valid_acc"], label="valid_acc")
    plt.title("feedback acc: {}".format(args.good_feedback_acc))
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(os.path.join(dir_name, "acc.jpg"), dpi=300)
    plt.close('all')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--reload_dir', type=str, default=None)

    parser.add_argument('--use_2D_env', action='store_true', default=USE_2D_ENV)
    parser.add_argument('--use_other_room', action='store_true', default=USE_OTHER_ROOM)
    parser.add_argument('--use_rich_reward', action='store_true', default=USE_RICH_REWARD)
    parser.add_argument('--use_multiple_starts', action='store_true', default=USE_MULTIPLE_STARTS)
    parser.add_argument('--pos_interval', type=float, default=POS_INTERVAL)
    parser.add_argument('--use_goal_info', action='store_true', default=USE_GOAL_INFO)
    parser.add_argument('--use_coords_and_orn', action='store_true', default=USE_COORDS_AND_ORN)
    parser.add_argument('--raycast_num', type=int, default=RAYCAST_NUM)
    parser.add_argument('--raycast_range', type=int, default=RAYCAST_RANGE)
    parser.add_argument('--goal_range', type=float, default=GOAL_RANGE)
    parser.add_argument('--start_varying_range', type=float, default=START_VARYING_RANGE)

    parser.add_argument('--use_feedback', action='store_true', default=USE_FEEDBACK)
    parser.add_argument('--use_real_feedback', action='store_true', default=USE_REAL_FEEDBACK)
    parser.add_argument('--good_feedback_acc', type=float, default=GOOD_FEEDBACK_ACC)
    parser.add_argument('--bad_feedback_acc', type=float, default=BAD_FEEDBACK_ACC)

    parser.add_argument('--nsteps', type=int, default=NSTEPS)
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIEMSTEPS)
    parser.add_argument('--only_use_hr_until', type=int, default=ONLY_USE_HR_UNTIL)
    parser.add_argument('--trans_to_rl_in', type=int, default=TRANS_TO_RL_IN)

    parser.add_argument('--ppo_lr', type=float, default=PPO_LR)
    parser.add_argument('--ent_coef', type=float, default=ENT_COEF)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--lam', type=float, default=LAMBDA)
    parser.add_argument('--cliprange', type=float, default=CLIPRANGE)
    parser.add_argument('--max_grad_norm', type=float, default=MAX_GRAD_NORM)
    parser.add_argument('--ppo_noptepochs', type=int, default=PPO_NOPTEPOCHS)
    parser.add_argument('--ppo_batch_size', type=int, default=PPO_BATCH_SIZE)
    parser.add_argument('--ppo_minibatch_size', type=int, default=PPO_MINIBATCH_SIZE)
    parser.add_argument('--init_rl_importance', type=float, default=INIT_RL_IMPORTANCE)

    parser.add_argument('--feedback_lr', type=float, default=FEEDBACK_LR)
    parser.add_argument('--feedback_noptepochs', type=int, default=FEEDBACK_NOPTEPOCHS)
    parser.add_argument('--feedback_batch_size', type=int, default=FEEDBACK_BATCH_SIZE)
    parser.add_argument('--feedback_minibatch_size', type=int, default=FEEDBACK_MINIBATCH_SIZE)
    parser.add_argument('--feedback_training_prop', type=float, default=FEEDBACK_TRAINING_PROP)
    parser.add_argument('--feedback_training_new_prop', type=float, default=FEEDBACK_TRAINING_NEW_PROP)
    parser.add_argument('--hf_loss_type', type=str, default=HF_LOSS_TYPE)
    parser.add_argument('--hf_loss_param', type=str, default=HF_LOSS_PARAM)
    parser.add_argument('--feedback_use_mixup', action='store_true', default=FEEDBACK_USE_MIXUP)
    parser.add_argument('--min_feedback_buffer_size', type=int, default=MIN_FEEDBACK_BUFFER_SIZE)

    parser.add_argument('--seed', type=int, default=SEED)

    args = parser.parse_args()

    train()
