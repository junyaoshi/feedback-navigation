import os
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from husky_sparse_vs_feedback_analysis import smooth, append_or_create_list_for_key

def get_args():
    parser = argparse.ArgumentParser(description='Process EEG from flanker task')

    parser.add_argument('-d', '--data_dir', type=str, default='../S',
                        help='Directory containing EEG data')
    parser.add_argument('-p', '--plots_dir', type=str, default='plots',
                        help='Directory to save plots')

    args = parser.parse_args()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    return args

def load_experiment_res(data_dir):
    predictions = []
    for path in glob.glob('{}/*'.format(data_dir)):
        print('Reading ' + path)
        data = pickle.load(open(path, 'rb'))
        prediction = data['pred']
        predictions.append(prediction)
    return np.concatenate(predictions)

def running_avg(list_to_avg, avg_steps=100):
    array_to_avg = np.asarray(list_to_avg)
    array_to_avg = array_to_avg.reshape(array_to_avg.shape[0], -1)
    array_cum_sum = np.copy(array_to_avg)
    for i in range(1, array_to_avg.shape[1]):
        array_cum_sum[:, i] = array_cum_sum[:, i - 1] + array_to_avg[:, i]

    array_avged = (array_cum_sum[:, avg_steps:] - array_cum_sum[:, :-avg_steps]) / avg_steps
    # array_avged = smooth(array_avged)

    return array_avged


def prediction_2_performance(predictions):
    avg_steps = 100
    label = predictions[:, 1]
    performance = np.copy(label)

    for i in range(1, len(performance)):
        performance[i] = performance[i - 1] + label[i]

    performance = smooth(performance, 0.1)
    avg_performance = (performance[avg_steps:] - performance[:-avg_steps]) / avg_steps
    # avg_performance = smooth(avg_performance, 0.1)
    return avg_performance


if __name__ == "__main__":
    args = get_args()
    predictions = load_experiment_res(args.data_dir)
    avg_performance = prediction_2_performance(predictions)

    hr_avg_steps = 100
    rl_avg_steps = 2000
    draw_hr_rl_in_same_subplot = True
    metrics_to_plot = ["feedback"]

    if not os.path.exists("plots"):
        os.mkdir("plots")

    performance_dict = {}
    for dirname in os.listdir("rslts"):
        dirname = os.path.join("rslts", dirname)
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
        if param["use_multiple_starts"]:
            continue

        # human reinforce
        if param["use_feedback"]:
            if not param["only_use_human_policy"]:
                continue
            assert param["good_feedback_acc"] == param["bad_feedback_acc"]
            key = "hr_acc_{}".format(param["good_feedback_acc"])
            append_or_create_list_for_key(performance_dict, key, performance)
        else:
            if param["use_sparse_reward"]:
                append_or_create_list_for_key(performance_dict, "rl_sparse", performance)
            else:
                append_or_create_list_for_key(performance_dict, "rl_rich", performance)

    for metric in metrics_to_plot:
        if draw_hr_rl_in_same_subplot:
            plt.figure(figsize=(8, 6))
            for key, val in performance_dict.items():
                avg_steps = hr_avg_steps if key[:2] == "hr" else rl_avg_steps
                val = running_avg([ele[metric] for ele in val], avg_steps)
                val_mean = val.mean(axis=0)
                val_std = val.std(axis=0)
                plt.plot(val_mean, label=key)
                plt.fill_between(range(len(val_mean)), val_mean + val_std, val_mean - val_std, alpha=0.2)
            plt.plot(avg_performance, label='experiment')

            plt.legend()
            plt.ylabel(metric)
            plt.xlabel("steps")
            plt.xlim((0, len(avg_performance)))
            plt.minorticks_on()
            plt.grid(True, which="both", alpha=.2)
            plt.savefig(os.path.join("plots", "{}.png".format(metric)))
            plt.show()
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 6))
            for key, val in performance_dict.items():
                is_hr_performance = key[:2] == "hr"
                avg_steps = hr_avg_steps if is_hr_performance else rl_avg_steps
                ax = ax1 if is_hr_performance else ax2
                val = running_avg([ele[metric] for ele in val], avg_steps)
                val_mean = val.mean(axis=0)
                val_std = val.std(axis=0)
                line, = ax.plot(val_mean, label=key)
                ax.fill_between(range(len(val_mean)), val_mean + val_std, val_mean - val_std, alpha=0.2)
                if not is_hr_performance:
                    _, ax1_xmax = ax1.get_xlim()
                    ax2.axhline(y=val_mean[int(ax1_xmax + hr_avg_steps)], color=line.get_color(), label="{}_@_1000_steps".format(key))

            ax1.plot(avg_performance, label='experiment')
            ax1.set_xlim((0, len(avg_performance)))
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
