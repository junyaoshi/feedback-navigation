from train_dqn import models  # noqa
from train_dqn.build_graph import build_act, build_train  # noqa
from train_dqn.deepq import load_act  # noqa
from train_dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
