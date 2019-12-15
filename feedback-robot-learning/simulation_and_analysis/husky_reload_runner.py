import subprocess
import os

if __name__ == "__main__":
    total_timesteps = int(5e3)
    trans_to_rl_in = int(2e10)
    init_rl_importance = 0.0
    reload_dir = os.path.join('simulation_and_analysis', 'rslts', '0905_Xiaomin_success', 'online')
    for i in range(1):
        for mode in ["feedback"]:
            arg = "python reload_husky_navigate_ppo2.py --seed {}".format(i)
            if "feedback" in mode:
                arg += " --use_feedback"
                arg += " --total_timesteps {}".format(total_timesteps)
                arg += " --trans_to_rl_in {}".format(trans_to_rl_in)
                arg += " --init_rl_importance {}".format(init_rl_importance)
                arg += " --reload_dir {}".format(reload_dir)
            else:
                raise ValueError("Unknown mode", mode)

            p = subprocess.Popen(arg, shell=True)
            p.communicate()
