import os
import json
import math
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
    array_to_avg = np.where(np.isnan(array_to_avg), 0, array_to_avg)
    array_cum_sum = np.copy(array_to_avg)
    for i in range(1, array_to_avg.shape[1]):
        array_cum_sum[:, i] = array_cum_sum[:, i - 1] + array_to_avg[:, i]

    array_avged = (array_cum_sum[:, avg_steps:] - array_cum_sum[:, :-avg_steps]) / avg_steps
    # array_avged = smooth(array_avged)

    return array_avged


def calculate_SPL(reward, avg_episode=20):
    res = []
    SPL = []
    last_goal = 0
    for i, r in enumerate(reward):
        if r == 0:
            traj_len = i - last_goal
            last_goal = i
            reached_goal = reward[i - 1] == 100
            SPL.append(reached_goal * 17 / max(traj_len, 17))
        latest_SPL = SPL[-avg_episode:] if len(SPL) else [0]
        res.append(np.mean(latest_SPL))

    return res


if __name__ == "__main__":
    acc_avg_steps = 100
    SPL_avg_steps = 250
    metrics_to_plot = ["train_acc", "valid_acc", "SPL"] # , "sparse_reward"
    parent_dir = os.path.join("rslts", "1022_log")

    if not os.path.exists("plots"):
        os.mkdir("plots")
    performance_dict = {}

    for dirname in os.listdir(parent_dir):
        dirname = os.path.join(parent_dir, dirname)
        if not os.path.isdir(dirname):
            continue
        param_file = os.path.join(dirname, "param.json")
        performance_file = os.path.join(dirname, "performance.p")
        if not os.path.exists(param_file) or not os.path.exists(performance_file):
            continue
        with open(param_file, "r") as f:
            param = json.load(f)
        with open(performance_file, "rb") as f:
            performance = pickle.load(f)

        if "SPL" in metrics_to_plot and "SPL" not in performance:
            performance["SPL"] = calculate_SPL(performance["sparse_reward"])
            with open(performance_file, "wb") as f:
                pickle.dump(performance, f)

        assert "hr" in dirname
        acc = param["good_feedback_acc"]
        loss_type = param["hf_loss_type"]
        loss_param = param["hf_loss_param"]
        if loss_type == "CCE":
            key = "CCE+0"
        else:
            key = "{}+{}".format(loss_type, loss_param)
        key += "+{}".format(acc)
        append_or_create_list_for_key(performance_dict, key, performance)
        print(dirname)

    for metric in metrics_to_plot:
        plt.figure(figsize=(18, 4))
        for i, acc in enumerate([0.7, 0.6, 0.55]):
            plt.subplot(1, 3, i + 1)
            for key, val_old in sorted(performance_dict.items()):
                loss_type, loss_param, task_acc = key.split("+")
                if task_acc != "{}".format(acc):
                    continue

                avg_steps = SPL_avg_steps if metric == "SPL" else acc_avg_steps
                val = running_avg([ele[metric] for ele in val_old], avg_steps)
                # print(val.shape)
                val_mean = val.mean(axis=0)
                val_std = val.std(axis=0) / 4

                line, = plt.plot(val_mean, label=loss_type)
                plt.fill_between(range(len(val_mean)),
                                 val_mean + val_std,
                                 val_mean - val_std, alpha=0.2)

            plt.title("Feedback Acc: {}".format(acc))
            plt.legend()
            plt.minorticks_on()
            plt.xlabel("steps")

        plt.savefig(os.path.join("plots", "hf_loss_{}.png".format(metric)))
        # plt.show()
