import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
import gym.spaces


class FeedbackPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, reuse=False, trans_by_interpolate=True):  # pylint: disable=W0613
        assert isinstance(ac_space, gym.spaces.Discrete)

        fb_constant = 8
        hidden_unit = 8
        l2_scale = 0.00

        ob_shape = (None,) + ob_space.shape
        actdim = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        Tmp = tf.placeholder(tf.float32, [], name='Tmp')
        RL_importance = tf.placeholder(tf.float32, (), name='RL_importance')

        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.layers.dense(X, hidden_unit, activation=tf.tanh,
                                 kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                 reuse=reuse, name='pi_fc1')
            pi = tf.layers.dense(h1, actdim, activation=None,
                                 kernel_initializer=tf.initializers.orthogonal(0.01),
                                 reuse=reuse, name='pi')
            h1 = tf.layers.dense(X, hidden_unit, activation=tf.tanh,
                                 kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                 reuse=reuse, name='vf_fc1')
            vf = tf.layers.dense(h1, 1, activation=None,
                                 kernel_initializer=tf.initializers.orthogonal(1.0),
                                 reuse=reuse, name='vf')[:, 0]
            h1 = tf.layers.dense(X, units=hidden_unit, activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale),
                                 reuse=reuse, name='feedback_fc1')
            fb = tf.layers.dense(h1, actdim, activation=None,
                                 kernel_initializer=tf.initializers.orthogonal(0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale),
                                 reuse=reuse, name='feedback')
            fb_soft2hard = tf.sigmoid(fb / Tmp)
            fb = tf.sigmoid(fb)

        self.pdtype = make_pdtype(ac_space)

        self.pd = self.pdtype.pdfromflat(pi)
        if trans_by_interpolate:
            rl_policy_logits = pi* RL_importance
            hr_policy_logits = fb * fb_constant * (1 - RL_importance)
            combined_logits = rl_policy_logits + tf.stop_gradient(hr_policy_logits) \
                              - tf.reduce_logsumexp(rl_policy_logits) \
                              - tf.stop_gradient(tf.reduce_logsumexp(hr_policy_logits))
        else:
            take_RL = tf.random.uniform((), 0, 1) < RL_importance
            combined_logits = tf.where(take_RL, pi, fb * fb_constant)
        self.combined_pd = self.pdtype.pdfromflat(combined_logits)

        a0 = self.combined_pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, rl_importance=0):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob, RL_importance: rl_importance})
            a = a[0]
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def feedback(action, n_act):
            indices = tf.stack([tf.range(n_act), action], axis=-1)
            return tf.gather_nd(fb, indices)

        def feedback_soft2hard(action):
            indices = tf.stack([tf.range(nbatch), action], axis=-1)
            return tf.gather_nd(fb_soft2hard, indices)

        self.X = X
        self.Tmp = Tmp
        self.pi = pi
        self.vf = vf
        self.fb = fb
        self.step = step
        self.value = value
        self.feedback = feedback
        self.feedback_soft2hard = feedback_soft2hard
