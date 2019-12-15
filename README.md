# Accelerated Robot Learning via Human Feedback

Purpose
=================
This project is developed as a final project for COMS E6998 Topics in Robot Learning

Contributors
=================
- Zizhao Wang: wangzizhao
- Jack Shi: junyaoshi

Introduction
=================

In reinforcement learning (RL), sparse rewards are a natural way to specify the task to be learned. However, most RL algorithms struggle to learn in this setting since the learning signal is mostly zeros. In contrast, humans are good at assessing and predicting the future consequences of actions and can serve as good reward/policy shapers to accelerate the robot learning process. In this work, we propose a method that uses evaluative feedback obtained from human to accelerate RL for robotic agents in sparse reward settings. As the robot learns the task, feedback of a human observer watching the robot attempts is recorded and decoded into noisy error feedback signal. From this feedback, we use supervised learning to obtain a policy that subsequently augments the behavior policy and guides exploration in the early stages of RL. This bootstraps the RL learning process to enable learning from sparse reward. Using a robotic navigation task as a test bed, we show that our method achieves a stable obstacle-avoidance policy with high success rate, outperforming learning from only sparse rewards that struggles to achieve obstacle avoidance behavior or fails to advance to the goal.

Installation of Gibson Environment
=================

We built the navigation task of the project in Gibson Environment. Please make sure to install Gibson Environment before running the code in this repository.

#### Installation Method

There are two ways to install gibson, A. using our docker image (recommended) and B. building from source. 

#### System requirements

The minimum system requirements are the following:

For docker installation (A): 
- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

For building from the source(B):
- Ubuntu >= 14.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 375
- CUDA >= 8.0, CuDNN >= v5


A. Quick installation (docker)
-----

You need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. 

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation. 

You can either 1. pull from our docker image (recommended) or 2. build your own docker image.


1. Pull from our docker image (recommended)

```bash
# download the dataset from https://storage.googleapis.com/gibsonassets/dataset.tar.gz
docker pull xf1280/gibson:0.3.1
xhost +local:root
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset xf1280/gibson:0.3.1
```

2. Build your own docker image 
```bash
git clone https://github.com/StanfordVL/GibsonEnv.git
cd GibsonEnv
./download.sh # this script downloads assets data file and decompress it into gibson/assets folder
docker build . -t gibson ### finish building inside docker, note by default, dataset will not be included in the docker images
xhost +local:root ## enable display from docker
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset gibson` to create a container. Note that we don't include
dataset files in docker image to keep our image slim, so you will need to mount it to the container when you start a container. 

#### Notes on deployment on a headless server

Gibson Env supports deployment on a headless server and remote access with `x11vnc`. 
You can build your own docker image with the docker file `Dockerfile` as above.
Instructions to run gibson on a headless server (requires X server running):

1. Install nvidia-docker2 dependencies following the starter guide. Install `x11vnc` with `sudo apt-get install x11vnc`.
2. Have xserver running on your host machine, and run `x11vnc` on DISPLAY :0.
3. `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset <gibson image name>`
4. Run gibson with `python <gibson example or training>` inside docker.
5. Visit your `host:5900` and you should be able to see the GUI.

If you don't have X server running, you can still run gibson, see [this guide](https://github.com/StanfordVL/GibsonEnv/wiki/Running-GibsonEnv-on-headless-server) for more details.

B. Building from source
-----
If you don't want to use our docker image, you can also install gibson locally. This will require some dependencies to be installed. 

First, make sure you have Nvidia driver and CUDA installed. If you install from source, CUDA 9 is not necessary, as that is for nvidia-docker 2.0. Then, let's install some dependencies:

```bash
apt-get update 
apt-get install libglew-dev libglm-dev libassimp-dev xorg-dev libglu1-mesa-dev libboost-dev \
		mesa-common-dev freeglut3-dev libopenmpi-dev cmake golang libjpeg-turbo8-dev wmctrl \
		xdotool libzmq3-dev zlib1g-dev
```	

Install required deep learning libraries: Using python3.5 is recommended. You can create a python3.5 environment first. 

```bash
conda create -n py35 python=3.5 anaconda 
source activate py35 # the rest of the steps needs to be performed in the conda environment
conda install -c conda-forge opencv
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
pip install torchvision==0.2.0
pip install tensorflow==1.3
```
Clone the repository, download data and build
```bash
git clone https://github.com/junyaoshi/bci-navigation.git
cd bci-navigation
./download.sh # this script downloads assets data file and decompress it into gibson/assets folder
./build.sh build_local ### build C++ and CUDA files
pip install -e . ### Install python libraries
```

Install OpenAI baselines if you need to run the training demo.

```bash
git clone https://github.com/fxia22/baselines.git
pip install -e baselines
```


Quick Start
=================

To run vanilla PPO training or Human-Guided RL

```bash
python3 feedback-robot-learning/husky_navigate_ppo2.py
```

Make sure to adjust the hyperparemeters in the file as needed.


To analyze experiment results

```bash
python3 feedback-robot-learning/simulation_and_analysis/husky_sparse_vs_feedback_analysis.py
```



