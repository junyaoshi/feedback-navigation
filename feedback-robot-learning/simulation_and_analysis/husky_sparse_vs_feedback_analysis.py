import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


def append_or_create_list_for_key(dict, key, ele):
    if key in dict:
        dict[key].append(ele)
    else:
        dict[key] = [ele]

def smooth(scalars, weight=0.9):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def running_avg(list_to_avg, avg_steps=100):
    array_to_avg = np.asarray(list_to_avg)
    array_to_avg = array_to_avg.reshape(array_to_avg.shape[0], -1)
    array_cum_sum = np.copy(array_to_avg)
    for i in range(1, array_to_avg.shape[1]):
        array_cum_sum[:, i] = array_cum_sum[:, i - 1] + array_to_avg[:, i]

    array_avged = (array_cum_sum[:, avg_steps:] - array_cum_sum[:, :-avg_steps]) / avg_steps
    # array_avged = smooth(array_avged)

    return array_avged


if __name__ == "__main__":
    only_use_hr_until = int(1e3)
    trans_to_rl_until = int(2e4) #+ only_use_hr_until
    hr_total_steps = int(4e4) # int(1e5) + only_use_hr_until
    hr_avg_steps = 1000
    rl_avg_steps = 2000
    avg_steps_hr_only = 100
    draw_hr_rl_in_same_subplot = True
    metrics_to_plot = ["feedback", "sparse_reward", "rich_reward"]
    parent_dir = os.path.join("rslts", "9.31 variant start 0.3 range coor & goal info 10 of 10 success")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    performance_dict = {}
    for dirname in os.listdir(parent_dir):
        dirname = os.path.join(parent_dir, dirname)
        if not os.path.isdir(dirname):
            continue
        param_file = os.path.join(dirname, "param.json")
        performance_file = os.path.join(dirname, "performance.p")
        model_dir = os.path.join(dirname, "models")
        if not os.path.exists(param_file) or not os.path.exists(performance_file):
            continue
        with open(param_file, "r") as f:
            # print('param file: {}'.format(param_file))
            param = json.load(f)
        if os.path.exists(model_dir):
            finished_steps = (len(os.listdir(model_dir)) - 2) * param["nsteps"] * 5
            trial_finished = param["total_timesteps"] == finished_steps
            # if not trial_finished:
            #     continue
        with open(performance_file, "rb") as f:
            performance = pickle.load(f)

        # if param['good_feedback_acc'] != 0.7:
        #     continue
        # if "reload_dir" not in param.keys():
        #     # print(dirname, param["use_real_feedback"])
        #     if param['use_feedback']:
        #         if not param["use_real_feedback"]:
        #             continue
        # if param["use_feedback"]:
        #     if param["only_use_rl_until"] != only_use_hr_until:
        #         continue

        # print(dirname, param["init_rl_importance"], param['only_use_hr_until'], np.mean(performance["feedback"][-100:]))
        # human reinforce
        if param["use_feedback"]:
            print(dirname, param["init_rl_importance"], param['only_use_hr_until'],
                  np.mean(performance["feedback"][-100:]))
            assert param["good_feedback_acc"] == param["bad_feedback_acc"]
            assert not param["use_rich_reward"]
            # key = "hr_{}_{}".format(param["trans_to_rl_in"], param["trans_by_interpolate"])
            # print('param_file: {}'.format(param_file))
            # key = "hr_init_rl_{}".format(param["init_rl_importance"])
            # key = "hr_initial_hr_only_steps_{}".format(param["only_use_hr_until"])
            # key = "hr_feedback_acc_0.7"
            # if param["use_real_feedback"]:
            #     key = "hr"
            # else:
            #     key = "hr_and_rl"
            # key = "hr_feedback_acc_{}".format(param["good_feedback_acc"])
            # if np.mean(performance["feedback"][-100:]) <= 0.43:
            #     continue
            # key = "hr_transition_to_rl_{}"

            key = "hr_varying_start_{}".format(param["only_use_hr_until"])
            append_or_create_list_for_key(performance_dict, key, performance)
        else:
            if not param["use_rich_reward"]:
                append_or_create_list_for_key(performance_dict, "rl_sparse", performance)
            else:
                append_or_create_list_for_key(performance_dict, "rl_rich", performance)
        # print(dirname, np.unique(performance['sparse_reward']))

    for metric in metrics_to_plot:
        if draw_hr_rl_in_same_subplot:
            plt.figure(figsize=(8, 6))
            for key, val_old in performance_dict.items():
                is_hr_performance = key[:2] == "hr" #and key != "hr_and_rl"
                avg_steps = hr_avg_steps if is_hr_performance else rl_avg_steps
                if is_hr_performance:
                    val_hr_only = running_avg([ele[metric] for ele in val_old], avg_steps_hr_only)
                    val_mean_hr_only = val_hr_only.mean(axis=0)
                    val_std_hr_only = val_hr_only.std(axis=0)
                    # line, = plt.plot(val_mean_hr_only, label=key)
                val = running_avg([ele[metric] for ele in val_old], avg_steps)
                val_mean = val.mean(axis=0)
                val_std = val.std(axis=0)
                line, = plt.plot(val_mean, label=key)
                if is_hr_performance:
                    # plt.fill_between(range(len(val_mean_hr_only)),
                    #                  val_mean_hr_only + val_std_hr_only,
                    #                  val_mean_hr_only - val_std_hr_only, alpha=0.2)
                    plt.fill_between(range(len(val_mean)),
                                     val_mean + val_std,
                                     val_mean - val_std, alpha=0.2)
                    plt.axhline(y=val_mean_hr_only[only_use_hr_until - avg_steps_hr_only], color=line.get_color(),
                                label="{}_@_{}_steps".format(key, only_use_hr_until))
                else:
                    plt.fill_between(range(only_use_hr_until, len(val_mean) + only_use_hr_until),
                                     val_mean + val_std, val_mean - val_std, alpha=0.2)
                    # plt.axvline(x=hr_total_steps, color='goldenrod',
                    #             label="performances_@_{}_steps".format(hr_total_steps))

            plt.legend()
            plt.ylabel(metric)
            plt.xlabel("steps")
            # plt.xlim((0, len(val_mean)))
            plt.xlim((0, len(val_mean)))
            plt.minorticks_on()
            plt.grid(True, which="both", alpha=.2)
            plt.savefig(os.path.join("plots", "{}.png".format(metric)))
            plt.show()
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 6))
            for key, val_old in performance_dict.items():
                is_hr_performance = key[:2] == "hr"
                avg_steps = hr_avg_steps if is_hr_performance else rl_avg_steps
                ax = ax1 if is_hr_performance else ax2
                val = running_avg([ele[metric] for ele in val_old], avg_steps)
                val_mean = val.mean(axis=0)
                val_std = val.std(axis=0)
                if is_hr_performance:
                    val_hr_only = running_avg([ele[metric] for ele in val_old], avg_steps_hr_only)
                    val_mean_hr_only = val_hr_only.mean(axis=0)
                line, = ax.plot(val_mean, label=key)
                ax.fill_between(range(len(val_mean)), val_mean + val_std, val_mean - val_std, alpha=0.2)
                if is_hr_performance:
                    _, ax1_xmax = ax1.get_xlim()
                    ax1.axhline(y=val_mean_hr_only[only_use_hr_until-avg_steps_hr_only], color=line.get_color(),
                                label="{}_@_{}_steps".format(key, only_use_hr_until))
                # else:
                #     _, ax1_xmax = ax1.get_xlim()
                #     ax2.axhline(y=val_mean[int(param['total_timesteps'])], color=line.get_color(),
                #                 label="{}_@_{}_steps".format(key, int(param['total_timesteps'])))
            ax1.set_xlabel("steps")
            ax1.set_ylabel(metric)
            ax1.set_title("human reinforcement")
            ax1.legend()
            ax2.set_xlabel("steps")
            ax2.set_ylabel(metric)
            ax2.set_title("reinforcement learning")
            ax2.legend()
            plt.savefig(os.path.join("plots", "{}.png".format(metric)))
            plt.show()
