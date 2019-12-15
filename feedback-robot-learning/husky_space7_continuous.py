# Import Libraries
from gibson.envs.husky_env import HuskyNavigateEnv
import inspect
from pprint import pprint
import pybullet as p
import os
import math
import random
import time as t
from time import sleep
from math import pi, degrees
# from transforms3d.euler import quat2euler

# Parse config and object arguments
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'husky_space7.yaml')
green_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'objects', 'green_cube.urdf')
red_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'objects', 'red_cube.urdf')
yellow_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'objects', 'yellow_cube.urdf')
blue_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'objects', 'blue_cube.urdf')
black_wall_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'objects', 'black_wall.urdf')
pprint('config file: {}'.format(config_file))
# pprint('white cube file: {}'.format(green_cube_file))
# pprint('blue cube file: {}'.format(blue_cube_file))

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default=config_file)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print(args)

# create environment
# env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
env = HuskyNavigateEnv(config=args.config, gpu_idx = args.gpu)
env.reset()

# (ord('s'), ): [-0.5,0], ## backward
# (ord('w'), ): [0.5,0], ## forward
# (ord('d'), ): [0,-0.5], ## turn right
# (ord('a'), ): [0,0.5], ## turn left

# define constants
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
IDLE = 4
initial_yaw = 4.71
initial_x, initial_y, initial_z = -14.3, 5, 1.2
target_x, target_y, target_z = env.robot.get_target_position()
wall1_y = 40
wall2_y = 0
_, _, _, _, _, _, _, _, cyaw, cpith, cdist, ctarget = p.getDebugVisualizerCamera()


def calc_3d_dist(p, q):
    """
    :param p: robot position list p
    :param q: robot position list q
    :return: the 3D Euclidean distance between p and q
    """
    return sum((p - q) ** 2 for p, q in zip(p, q)) ** 0.5


def calc_2d_dist(p, q):
    """
    :param p: robot position list p
    :param q: robot position list q
    :return: the 2D Euclidean distance between p and q
    """
    p = [p[0], p[1]]
    q = [q[0], q[1]]
    return sum((p - q) ** 2 for p, q in zip(p, q)) ** 0.5


def calc_x_dist(p, q):
    """
    :param p: robot position list p
    :param q: robot position list q
    :return: the x-axis Euclidean distance between p and q
    """
    return abs(p[0]-q[0])


def quaternion_to_euler(orn, yaw_only=True):
    """
    :param orn: a 4-element list of quarternion angles
    :return: a 3-element list of eulerian angles [yaw, pitch, roll]
    """
    x, y, z, w = orn

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    if yaw_only:
        return yaw
    else:
        return [yaw, pitch, roll]


def convert_angle(angle):
    """
    convert angle to range (0, 2*pi)
    :param angle:
    :return: converted angle
    """
    if angle > 2*pi:
        angle -= 2*pi
    elif angle < 0:
        angle += 2*pi
    return angle


def brake(env, time=100, adjustment_precision=0.0001):
    """
    Brakes
    :param adjustment_precision: the amount of adjustment the robot accounts for its drifting
    """
    print('Braking for {} timesteps'.format(time))
    y_pos = env.robot.get_position()[1]
    for i in range(time):
        ensure_orientation(env)
        new_y_pos = env.robot.get_position()[1]
        if new_y_pos < y_pos:
            env.robot.move_backward(adjustment_precision)
        elif new_y_pos > y_pos:
            env.robot.move_forward(adjustment_precision)
        y_pos = new_y_pos
        env.step(IDLE)


def brake_at_state(env, state, time=100, adjustment_precision=0.001):
    """
    Brakes
    :param adjustment_precision: the amount of adjustment the robot accounts for its drifting
    """
    print('Braking at state {} for {} timesteps'.format(state, time))
    for i in range(time):
        ensure_orientation(env)
        x_pos = env.robot.get_position()[0]
        if x_pos < state:
            env.robot.move_backward(adjustment_precision)
        elif x_pos > state:
            env.robot.move_forward(adjustment_precision)
        env.step(IDLE)


def ensure_orientation(env, tolerance=0.01):
    """
    Ensures the robot's orientation does not change
    """
    yaw = convert_angle(quaternion_to_euler(env.robot.get_orientation()))
    while abs(yaw - initial_yaw) > tolerance:
        if yaw < initial_yaw:
            env.step(LEFT)
        else:
            env.step(RIGHT)
        yaw = convert_angle(quaternion_to_euler(env.robot.get_orientation()))


def ensure_fov(env):
    """
    Ensures that the field of view is consistent with robot's orientation
    """
    yaw_radian = convert_angle(quaternion_to_euler(env.robot.get_orientation()))
    yaw_degree = degrees(yaw_radian)
    print('yaw: ', yaw_degree)
    p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=yaw_degree-90, cameraPitch=cpith, cameraTargetPosition=ctarget)


def move_forward(env, distance=3, enforce_state=True):
    """
    Moves robot forward
    """
    print("Moving forward for {} meters".format(abs(distance)))
    p = env.robot.get_position()
    q = p
    if enforce_state:
        x_pos = p[0]
        current_state = int(round(x_pos))
        goal_state = current_state - distance
        while x_pos > goal_state:
            ensure_orientation(env)
            _, rew, _, _ = env.step(FORWARD)
            x_pos = env.robot.get_position()[0]
    else:
        while calc_x_dist(p, q) <= distance:
            ensure_orientation(env)
            _, rew, _, _ = env.step(FORWARD)
            q = env.robot.get_position()


def move_backward(env, distance=3, enforce_state=True):
    """
    :param enforce_state: ensures that the robot arrives at a dicrete state
    Moves robot backward
    """
    print("Moving backward for {} meters".format(abs(distance)))
    p = env.robot.get_position()
    q = p
    if enforce_state:
        x_pos = p[0]
        current_state = int(round(x_pos))
        goal_state = current_state + distance
        while x_pos < goal_state:
            ensure_orientation(env)
            _, rew, _, _ = env.step(BACKWARD)
            x_pos = env.robot.get_position()[0]
    else:
        while calc_x_dist(p, q) <= distance:
            ensure_orientation(env)
            _, rew, _, _ = env.step(BACKWARD)
            q = env.robot.get_position()


def take_action(env, action, time=100, indicator_file=None):
    """
    take action for designated timesteps
    """
    phrase = None
    chosen_action = None
    if action == 'forward':
        phrase = 'Accelerating forward'
        chosen_action = FORWARD
    elif action == 'backward':
        phrase = 'Accelerating backward'
        chosen_action = BACKWARD
    elif action == 'idle':
        phrase = 'Staying idle'
        chosen_action = IDLE
    else:
        print('Action input unrecognized! Please use input [forward], [backward], or [idle]!')
        return False

    print("{} for {} timesteps".format(phrase, time))
    pos = env.robot.get_position()

    # load initial indicator object
    x0, y0, z0 = pos
    if indicator_file is not None:
        id = p.loadURDF(fileName=indicator_file,
                        basePosition=[x0, y0, z0 + 1],
                        useFixedBase=True)

    # move robot while generating indicator object
    for i in range(time):
        pos = env.robot.get_position()
        orn = env.robot.get_orientation()
        x, y, z = pos

        # check if robot has reached goal
        if goal_reached(y):
            brake(env, time=5)
            if indicator_file is not None:
                p.removeBody(id)
            random_y = random.uniform(wall1_y, wall2_y)
            print('Goal reached! Reset to random new location: {}'.format(random_y))
            env.robot.reset_new_pose([x, random_y, z], orn)
            return True

        # generate indicator object at the begining
        if i == 0 and indicator_file is not None:
            p.resetBasePositionAndOrientation(bodyUniqueId=id, posObj=[x, y, z + 1], ornObj=orn)

        print(i)
        if i == 3 and indicator_file is not None:
            p.removeBody(id)

        # move robot
        ensure_orientation(env)
        env.step(chosen_action)

    return False


def goal_reached(y, tolerance=0.1):
    return abs(y - target_y) <= tolerance


def main():
    timestep = 10000

    p.loadURDF(fileName=blue_cube_file,
               basePosition=[target_x, target_y, target_z],
               useFixedBase=True)

    p.loadURDF(fileName=black_wall_file,
               basePosition=[target_x, wall1_y, target_z],
               useFixedBase=True)

    p.loadURDF(fileName=black_wall_file,
               basePosition=[target_x, wall2_y, target_z],
               useFixedBase=True)

    # robot takes random actions
    brake(env, time=50)
    ensure_fov(env)
    for i in range(timestep):
        print('robot position: {}'.format(env.robot.get_position()))
        print('robot orientation: {}'.format(quaternion_to_euler(env.robot.get_orientation())))
        print('robot distance to target: {}'.format(env.robot.dist_to_target()))
        num = random.random()
        object_id = 0
        if num < 1/3:
            take_action(env, action='forward', time=30, indicator_file=green_cube_file)
        elif num < 2/3:
            take_action(env, action='idle', time=30, indicator_file=yellow_cube_file)
        else:
            take_action(env, action='backward', time=30, indicator_file=red_cube_file)
        timestep += 1

    brake(env, time=400)
    env.close()


if __name__ == '__main__':
    main()
    # print('sensor space: {}\n observation space: {}\n action space: {}'.format(env.sensor_space, env.observation_space, env.action_space))






