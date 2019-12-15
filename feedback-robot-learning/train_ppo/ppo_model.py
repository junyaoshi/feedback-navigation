import joblib
import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space,
                 nbatch_act, nbatch_train, nbatch_feedback,
                 nsteps, ent_coef, vf_coef, max_grad_norm,
                 hf_loss_type="CCE", hf_loss_param=None):
        sess = tf.get_default_session()
        act_model = policy(sess, ob_space, ac_space, nbatch_act, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, reuse=True)
        feedback_model = policy(sess, ob_space, ac_space, nbatch_feedback, reuse=True)

        # PPO placeholder
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        # PPO loss and train_op
        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # grads_4_print = tf.gradients(loss, params)
        # grads_4_print = [g for g in grads_4_print if g is not None]
        # print('grads_print: {}'.format(grads_4_print))

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        train_op = trainer.apply_gradients(grads)

        # feedback placeholder
        FB = tf.placeholder(tf.float32, [None], name="FB")
        A_FB = tf.placeholder(tf.int32, [None], name="A_FB")
        LR_FB = tf.placeholder(tf.float32, [], name="LR_FB")

        W = tf.placeholder(tf.float32, [None], name="W")
        BOOTSTRAP = tf.placeholder(tf.bool, [], name="BOOTSTRAP")
        N_ACT = tf.placeholder(tf.int32, [], name="N_ACT")

        # feedback loss and train_op
        pred_feedbacks = tf.clip_by_value(feedback_model.feedback(A_FB, N_ACT), 1e-6, 1 - 1e-6)
        CCE = -(FB * tf.log(pred_feedbacks) + (1 - FB) * tf.log(1 - pred_feedbacks))
        if hf_loss_type == "CCE":
            fb_loss = tf.reduce_mean(CCE)
        elif hf_loss_type == "CCE_ABN":
            good_acc, bad_acc = hf_loss_param
            positive_prob = good_acc * pred_feedbacks + (1 - bad_acc) * (1 - pred_feedbacks)
            negative_prob = (1 - good_acc) * pred_feedbacks + bad_acc * (1 - pred_feedbacks)
            fb_loss = -tf.reduce_mean(FB * tf.log(positive_prob) + (1 - FB) * tf.log(negative_prob))
        elif hf_loss_type == "soft_boot":
            soft_lambda = hf_loss_param[0]
            ENT = -(pred_feedbacks * tf.log(pred_feedbacks) + (1 - pred_feedbacks) * tf.log(1 - pred_feedbacks))
            fb_loss = tf.reduce_mean(soft_lambda * CCE + (1 - soft_lambda) * ENT)
        elif hf_loss_type == "hard_boot":
            hard_lambda = hf_loss_param[0]
            ENT = -(tf.where(tf.greater(pred_feedbacks, 0.5), tf.log(pred_feedbacks), tf.log(1 - pred_feedbacks)))
            fb_loss = tf.reduce_mean(hard_lambda * CCE - (1 - hard_lambda) * ENT)
        elif hf_loss_type == "Lq":
            q, k = hf_loss_param
            pred_prob = tf.where(tf.equal(FB, 1), pred_feedbacks, 1 - pred_feedbacks)
            pred_prob = tf.clip_by_value(pred_prob, k, 1)
            fb_loss = tf.reduce_mean((1 - (pred_prob + 1e-6) ** q) / q)
        elif hf_loss_type == "SL":
            alpha, A = hf_loss_param
            RCCE = -tf.where(tf.equal(FB, 0), A * pred_feedbacks, A * (1 - pred_feedbacks))
            fb_loss = tf.reduce_mean(alpha * CCE + RCCE)
        elif "bmm" in hf_loss_type:
            stage_loss = CCE
            fb_soft2hard = feedback_model.feedback_soft2hard(A_FB)
            fb_soft2hard = tf.clip_by_value(fb_soft2hard, 1e-6, 1 - 1e-6)
            ENT = -(fb_soft2hard * tf.log(pred_feedbacks) + (1 - fb_soft2hard) * tf.log(1 - pred_feedbacks))
            fb_loss = tf.reduce_mean((1 - W) * stage_loss + W * ENT)
            if hf_loss_type == "bmm":
                mean_pred_feedback = tf.reduce_mean(pred_feedbacks)
                reg = 0.5 * tf.log(0.5 / mean_pred_feedback) + 0.5 * tf.log(0.5 / (1 - mean_pred_feedback))
                fb_loss = tf.where(BOOTSTRAP, fb_loss + 1 * reg, fb_loss)
        else:
            raise ValueError

        # feedback_trainer = tf.train.AdamOptimizer(learning_rate=LR_FB)
        # fb_loss += tf.losses.get_regularization_loss()
        feedback_trainer = tf.train.GradientDescentOptimizer(learning_rate=LR_FB)
        feedback_train_op = feedback_trainer.minimize(fb_loss)

        params = tf.trainable_variables()

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            # ratio_val, grad_val = sess.run([ratio, grads_4_print], td_map)
            # print('advs: {}'.format(advs))
            # print('ratio min: {}, ratio max: {}'.format(ratio_val.min(), ratio_val.max()))
            # print('ratio_val: {}'.format(ratio_val))
            # print('grad min: {}, grad max: {}'.format(min([g.min() for g in grad_val]), max([g.max() for g in grad_val])))
            # print('grad_val: {}'.format(grad_val))
            # if ratio_val.min() == 1.0 and ratio_val.max == 1.0:
            #     pass
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, train_op],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def feedback_train(lr, obs, actions, feedbacks):
            td_map = {feedback_model.X: obs, A_FB: actions, FB: feedbacks, LR_FB: lr, N_ACT: nbatch_feedback}
            return sess.run([pred_feedbacks, fb_loss, feedback_train_op], td_map)

        def feedback_train_bootstrap(lr, obs, actions, feedbacks, bmm_model, use_bootstrap=False, tmp=1.0):
            bootstrap = False
            w = np.zeros(nbatch_feedback)
            if use_bootstrap:
                bootstrap = True
                batch_CCE = sess.run(CCE, {feedback_model.X: obs, A_FB: actions, FB: feedbacks, N_ACT: nbatch_feedback})
                w = bmm_model.look_lookup(batch_CCE)

            td_map = {feedback_model.X: obs, A_FB: actions, LR_FB: lr, FB: feedbacks,
                      feedback_model.Tmp: tmp, W: w, BOOTSTRAP: bootstrap, N_ACT: nbatch_feedback}
            return sess.run([pred_feedbacks, fb_loss, feedback_train_op], td_map)

        def feedback_evaluate(obs, actions, feedbacks):
            assert len(obs) == len(actions) == len(feedbacks)
            total_acc = []
            total_loss = []
            for i in range(0, len(obs), nbatch_feedback):
                obs_batch = obs[i:i + nbatch_feedback]
                action_batch = actions[i:i + nbatch_feedback]
                feedback_batch = feedbacks[i:i + nbatch_feedback]
                td_map = {feedback_model.X: obs_batch, A_FB: action_batch, FB: feedback_batch, N_ACT: len(feedback_batch)}
                pred, loss = sess.run([pred_feedbacks, CCE], td_map)
                total_acc = np.concatenate([total_acc, np.round(pred) == feedback_batch])
                total_loss = np.concatenate([total_loss, loss])
            return total_acc, total_loss

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.feedback_train = feedback_train
        self.feedback_train_bootstrap = feedback_train_bootstrap
        self.feedback_evaluate = feedback_evaluate
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101
