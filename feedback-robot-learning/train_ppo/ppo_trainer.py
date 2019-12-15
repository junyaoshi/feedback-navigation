import os
import time
import math
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance

import pylsl
import pickle
from train_ppo.ppo_model import Model
from train_ppo.trainer_helper import run_dijkstra, judge_action_1D, get_simulated_feedback, get_feedback_from_LSL
from train_ppo.BMM import train_bmm_model


class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam,
                 judge_action=None,
                 use_rich_reward=False,
                 use_multiple_starts=False,
                 use_feedback=False,
                 use_real_feedback=False,
                 only_use_hr_until=1000,
                 trans_to_rl_in=1000,
                 init_rl_importance=0.2):
        self.env = env
        self.model = model
        nenv = 1

        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs_sensor = np.zeros((nenv,) + env.sensor_space.shape, dtype=model.train_model.X.dtype.name)
        obs_all = self.env.reset()

        self.obs_sensor[:] = obs_all['nonviz_sensor']
        self.obs[:] = obs_all['obs']
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = False
        self.judge_action = judge_action

        self.use_rich_reward = use_rich_reward
        self.use_multiple_starts = use_multiple_starts
        self.use_feedback = use_feedback
        self.use_real_feedback = use_real_feedback
        self.only_use_hr_until = only_use_hr_until
        self.trans_to_rl_in = trans_to_rl_in
        self.init_rl_importance = init_rl_importance
        self.rl_importance = 0
        self.num_step = 0

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []

        mb_action_idxs, mb_cors = [], []
        mb_sparse_rew, mb_rich_rew = [], []

        for _ in range(self.nsteps):
            self.num_step += 1
            self.env.env.action_idx = self.num_step

            if self.use_feedback:
                # transfer from hr policy to rl policy
                # between steps [only_use_hr_until, only_use_hr_until + trans_to_rl_in]
                # rl_importance = (self.num_step - self.only_use_hr_until) / self.trans_to_rl_in

                if self.dones:
                    hr_num_step = self.only_use_hr_until + self.trans_to_rl_in
                    if self.num_step <= self.only_use_hr_until:
                        self.rl_importance = 0
                    elif self.num_step <= hr_num_step:
                        self.env.env.episode_max_len = 120
                        self.rl_importance = np.random.uniform(0, 1) < \
                                             (self.num_step - self.only_use_hr_until) / self.trans_to_rl_in * \
                                             (1 - self.init_rl_importance) + self.init_rl_importance
                    else:
                        self.rl_importance = 1
                self.rl_importance = np.clip(self.rl_importance, 0, 1)
                # print('rl importance: {}'.format(self.rl_importance))
            else:
                self.rl_importance = 1

            if self.env.env.config['offline']:
                good_actions, bad_actions = [], []
                for action in range(self.env.action_space.n):
                    if self.judge_action(self.obs_sensor[0], action):
                        good_actions.append(action)
                    else:
                        bad_actions.append(action)
                if not bad_actions or not good_actions:
                    actions = np.random.choice(self.env.action_space.n)
                else:
                    actions = np.random.choice(np.random.choice([good_actions, bad_actions]))
                _, values, self.states, neglogpacs = self.model.step(self.obs)
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, self.rl_importance)
                if self.use_feedback and self.use_real_feedback and self.num_step < self.only_use_hr_until: #
                    if np.random.uniform() < 0.1 + 0.1 * (1 - self.num_step / self.only_use_hr_until):
                        actions = 0

            mb_obs.append(self.obs.copy())
            mb_actions.append([actions])
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append([self.dones])

            mb_action_idxs.append(self.num_step)
            mb_cors.append(self.obs_sensor.copy())

            if self.dones:
                obs_all = self.env.reset()
                if self.use_multiple_starts:
                    obs_all = self.env.env.random_reset()
                self.obs_sensor[:] = obs_all['nonviz_sensor']
                self.obs[:] = obs_all['obs']
                rewards = sparse_rewards = rich_rewards = 0
                self.dones = False
            else:
                obs_all, rewards_dict, self.dones, infos = self.env.step(actions)

                sparse_rewards = rewards_dict["sparse"]
                rich_rewards = rewards_dict["rich"]

                self.obs_sensor[:] = obs_all['nonviz_sensor']
                self.obs[:] = obs_all['obs']
                if infos['episode'] is not None:
                    maybeepinfo = infos['episode']
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)

                rewards = rich_rewards if self.use_rich_reward else sparse_rewards

            mb_rewards.append([rewards])

            mb_sparse_rew.append([sparse_rewards])
            mb_rich_rew.append([rich_rewards])
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        mb_cors = np.asarray(mb_cors, dtype=self.obs_sensor.dtype)
        mb_sparse_rew = np.asarray(mb_sparse_rew, dtype=np.float32)
        mb_rich_rew = np.asarray(mb_rich_rew, dtype=np.float32)

        return (*map(sf01, (mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_cors, mb_sparse_rew, mb_rich_rew)),
                mb_states, mb_action_idxs, epinfos)

    def calculate_returns(self, mb_rewards, mb_dones, mb_values):
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        nsteps = len(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones)
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return mb_returns


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    # swap and then flatten axes 0 and 1
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, raw_env,
          use_2D_env=True,
          use_other_room=False,
          use_rich_reward=False,
          use_multiple_starts=False,
          use_feedback=True,
          use_real_feedback=False,
          only_use_hr_until=1000,
          trans_to_rl_in=1000,
          nsteps=8,
          total_timesteps=1000,
          ppo_lr=2e-4, cliprange=0.2, ent_coef=.1, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          ppo_noptepochs=4, ppo_batch_size=32, ppo_minibatch_size=8, init_rl_importance=0.2,
          feedback_lr=1e-3, min_feedback_buffer_size=32,
          feedback_noptepochs=4, feedback_batch_size=16, feedback_minibatch_size=8,
          feedback_training_prop=0.7,
          feedback_training_new_prop=0.4,
          feedback_use_mixup=False,
          hf_loss_type="CCE", hf_loss_param=None,
          good_feedback_acc=0.7,
          bad_feedback_acc=0.7,
          log_interval=10, save_interval=0, reload_name=None, base_path=None):

    if isinstance(ppo_lr, float):
        ppo_lr = constfn(ppo_lr)
    else:
        assert callable(ppo_lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    total_timesteps = int(total_timesteps)
    assert ppo_batch_size % nsteps == 0

    ob_space = env.observation_space
    ac_space = env.action_space

    nenvs = 1
    nbatch = nenvs * nsteps

    if hf_loss_type == 0:
        hf_loss_param = None


    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                               nbatch_act=nenvs, nbatch_train=ppo_minibatch_size, nbatch_feedback=feedback_minibatch_size,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm,
                               hf_loss_type=hf_loss_type,
                               hf_loss_param=hf_loss_param)

    if save_interval and logger.get_dir():
        import cloudpickle
        if not base_path:
            base_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(osp.join(base_path, "models")):
            os.mkdir(osp.join(base_path, "models"))
        with open(osp.join(base_path, "models", 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    if use_real_feedback:
        print("looking for an EEG_Pred stream...", end="", flush=True)
        feedback_LSL_stream = pylsl.StreamInlet(pylsl.resolve_stream('type', 'EEG_Pred')[0])
        print(" done")

    model = make_model()
    if reload_name:
        model.load(reload_name)

    target_position = raw_env.robot.get_target_position()
    if use_2D_env:
        judge_action, *_ = run_dijkstra(raw_env, target_position, use_other_room=use_other_room)
    else:
        judge_action = judge_action_1D(raw_env, target_position)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                    judge_action=judge_action,
                    use_rich_reward=use_rich_reward,
                    use_multiple_starts=use_multiple_starts,
                    use_feedback=use_feedback,
                    use_real_feedback=use_real_feedback,
                    only_use_hr_until=only_use_hr_until,
                    trans_to_rl_in=trans_to_rl_in,
                    init_rl_importance=init_rl_importance)

    epinfobuf = deque(maxlen=100)

    nupdates = total_timesteps // nbatch

    state_action_buffer = deque(maxlen=100)
    action_idx_buffer = deque(maxlen=100)

    feedback_buffer_train = {}
    feedback_buffer_train_true = {}
    feedback_buffer_valid = {}
    feedback_bmms = {}
    for a in range(ac_space.n):
        feedback_buffer_train[a], feedback_buffer_train_true[a], feedback_buffer_valid[a] = [], [], []
        feedback_bmms[a] = 0
    performance = {"feedback": [], "sparse_reward": [], "rich_reward": [],
                   "train_acc": [], "train_true_acc": [], "valid_acc": []}
    epi_test_num = [0 for _ in range(ac_space.n)]

    ppo_obs, ppo_rewards, ppo_masks, ppo_actions, ppo_values, ppo_neglogpacs = [], [], [], [], [], []
    for update in range(1, nupdates + 1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        ppo_lrnow = ppo_lr(frac)
        cliprangenow = cliprange(frac)

        obs, rewards, masks, actions, values, neglogpacs, cors, sparse_rew, rich_rew, _, action_idxs, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        performance["sparse_reward"].extend(sparse_rew)
        performance["rich_reward"].extend(rich_rew)

        mblossvals = []

        state_action_buffer.extend([[s, a] for s, a in zip(obs, actions)])
        action_idx_buffer.extend(action_idxs)

        if use_feedback:
            if use_real_feedback:
                action_idxs, feedbacks, correct_feedbacks = get_feedback_from_LSL(feedback_LSL_stream)
                print("Received feedback from LSL", feedbacks)
            else:
                action_idxs, feedbacks, correct_feedbacks = \
                    get_simulated_feedback(cors if use_2D_env else obs, actions, action_idxs, judge_action,
                                           good_feedback_acc, bad_feedback_acc)
            performance["feedback"].extend(correct_feedbacks)

            # add feedbacks into feedback replay buffer
            if len(feedbacks):
                for a_idx, fb, cfb in zip(action_idxs, feedbacks, correct_feedbacks):
                    s, a = state_action_buffer[action_idx_buffer.index(a_idx)]
                    epi_test_num[a] += 1 - feedback_training_prop
                    # s, fb, cfb = np.ones(13), 1, 1
                    if epi_test_num[a] > 1:
                        feedback_buffer_valid[a].append([s, cfb])
                        epi_test_num[a] -= 1
                    else:
                        feedback_buffer_train[a].append([s, fb])
                        feedback_buffer_train_true[a].append([s, cfb])


        # train PPO
        if runner.num_step >= only_use_hr_until:
            ppo_obs.extend(obs)
            ppo_rewards.extend(rewards)
            ppo_masks.extend(masks)
            ppo_actions.extend(actions)
            ppo_values.extend(values)
            ppo_neglogpacs.extend(neglogpacs)

            if len(ppo_obs) == ppo_batch_size:
                ppo_obs = np.asarray(ppo_obs)
                ppo_rewards = np.asarray(ppo_rewards)
                ppo_masks = np.asarray(ppo_masks)
                ppo_actions = np.asarray(ppo_actions)
                ppo_values = np.asarray(ppo_values)
                ppo_neglogpacs = np.asarray(ppo_neglogpacs)
                ppo_returns = runner.calculate_returns(ppo_rewards, ppo_masks, ppo_values)
                inds = np.arange(ppo_batch_size)
                for _ in range(ppo_noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, ppo_batch_size, ppo_minibatch_size):
                        end = start + ppo_minibatch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds]
                                  for arr in (ppo_obs, ppo_returns, ppo_masks, ppo_actions, ppo_values, ppo_neglogpacs))
                        mblossvals.append(model.train(ppo_lrnow, cliprangenow, *slices))
                ppo_obs, ppo_rewards, ppo_masks, ppo_actions, ppo_values, ppo_neglogpacs = [], [], [], [], [], []

        # train feedback regressor
        if use_feedback and runner.num_step <= only_use_hr_until:
            all_train_acc = []
            all_train_true_acc = []
            all_valid_acc = []

            if not all([len(feedback_buffer) >= min_feedback_buffer_size
                        for feedback_buffer in feedback_buffer_train.values()]):
                performance["train_acc"].append(0.)
                performance["train_true_acc"].append(0.)
                performance["valid_acc"].append(0.)
                continue
            for a in range(ac_space.n):
                feedback_buffer = feedback_buffer_train[a]
                feedback_buffer_t = feedback_buffer_train_true[a]
                feedback_buffer_v = feedback_buffer_valid[a]
                bmm_model = feedback_bmms[a]
                for i in range(feedback_noptepochs):
                    # print(len(feedback_buffer))
                    # print(feedback_buffer[:3])
                    if i < feedback_noptepochs * feedback_training_new_prop:
                        inds = np.arange(len(feedback_buffer) - feedback_batch_size, len(feedback_buffer))
                    else:
                        inds = np.random.choice(len(feedback_buffer), feedback_batch_size, replace=False)

                    np.random.shuffle(inds)
                    for start in range(0, feedback_batch_size, feedback_minibatch_size):
                        end = start + feedback_minibatch_size
                        obs       = np.asarray([feedback_buffer[idx][0] for idx in inds[start:end]])
                        feedbacks = np.asarray([feedback_buffer[idx][1] for idx in inds[start:end]])
                        actions   = np.asarray([a] * feedback_minibatch_size)
                        if "bmm" in hf_loss_type:
                            prop1, prop2 = hf_loss_param
                            use_bootstrap = update * nsteps > only_use_hr_until * prop1
                            tmp = 1 - (1 - 0.001) * (update * nsteps - use_bootstrap) / (only_use_hr_until * prop2 - use_bootstrap)
                            tmp = min(tmp, 0.001)
                            pred, loss, _ = \
                                model.feedback_train_bootstrap(feedback_lr, obs, actions, feedbacks, bmm_model,
                                                               use_bootstrap, tmp)
                        else:
                            pred, loss, _ = model.feedback_train(feedback_lr, obs, actions, feedbacks)
                        # print('action: {} feedback: {} pred: {} loss: {}'.format(actions, feedbacks, pred, loss))

                evaluate_start = time.time()
                obs_train       = np.array([ele[0] for ele in feedback_buffer])
                feedbacks_train = np.array([ele[1] for ele in feedback_buffer])
                actions_train   = np.array([a] * len(feedback_buffer))

                obs_valid       = np.array([ele[0] for ele in feedback_buffer_v])
                feedbacks_valid = np.array([ele[1] for ele in feedback_buffer_v])
                actions_valid   = np.array([a] * len(feedback_buffer_v))

                obs_train_true       = np.array([ele[0] for ele in feedback_buffer_t])
                feedbacks_train_true = np.array([ele[1] for ele in feedback_buffer_t])
                actions_train_true   = np.array([a] * len(feedback_buffer_t))

                train_acc, train_loss = model.feedback_evaluate(obs_train, actions_train, feedbacks_train)
                valid_acc, _ = model.feedback_evaluate(obs_valid, actions_valid, feedbacks_valid)
                train_true_acc, _ = model.feedback_evaluate(obs_train_true, feedbacks_train_true, actions_train_true)

                feedback_bmms[a] = train_bmm_model(train_loss,
                                                   a, update, base_path, feedbacks_train == feedbacks_train_true, good_feedback_acc)

                all_train_acc = np.concatenate([all_train_acc, train_acc])
                all_valid_acc = np.concatenate([all_valid_acc, valid_acc])
                all_train_true_acc = np.concatenate([all_train_true_acc, train_true_acc])
                # print("evaluation takes ", time.time() - evaluate_start)

            all_train_acc, all_train_true_acc, all_valid_acc = \
                np.mean(all_train_acc), np.mean(all_train_true_acc), np.mean(all_valid_acc)
            print("train acc {:>4.2f}; train true acc {:>4.2f}; valid acc {:>4.2f}".format(
                all_train_acc, all_train_true_acc, all_valid_acc))
            performance["train_acc"].append(all_train_acc if math.isfinite(all_train_acc) else 0.)
            performance["train_true_acc"].append(all_train_true_acc if math.isfinite(all_train_true_acc) else 0.)
            performance["valid_acc"].append(all_valid_acc if math.isfinite(all_valid_acc) else 0.)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            # logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            # logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            # logger.logkv('time_elapsed', tnow - tfirststart)
            # for (lossval, lossname) in zip(lossvals, model.loss_names):
            #     logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            model_dir = osp.join(base_path, "models")
            os.makedirs(model_dir, exist_ok=True)
            savepath = osp.join(model_dir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            print("Saved model successfully.")
            if use_feedback:
                performance_fname = os.path.join(base_path, "performance.p")
                with open(performance_fname, "wb") as f:
                    pickle.dump(performance, f)
    env.close()

    return performance


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def enjoy(*, policy, env, total_timesteps, base_path,
          ent_coef=.1, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          nsteps=8, ppo_minibatch_size=16, feedback_minibatch_size=4):

    total_timesteps = int(total_timesteps)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                               nbatch_act=nenvs, nbatch_train=ppo_minibatch_size, nbatch_feedback=feedback_minibatch_size,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm,
                               trans_by_interpolate=False)

    model = make_model()
    model_dir = osp.join(base_path, 'models')
    max_model_iter = -1
    for fname in os.listdir(model_dir):
        if fname.isdigit():
            model_iter = int(fname)
            if model_iter > max_model_iter:
                max_model_iter = model_iter
                reload_name = osp.join(model_dir, fname)

    if reload_name:
        model.load(reload_name)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                    use_feedback=False)

    for update in range(1, total_timesteps + 1):
        runner.run()  # pylint: disable=E0632

    env.close()