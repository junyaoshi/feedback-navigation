from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Husky
from transforms3d import quaternions
import os
from gym import spaces
from math import pi, degrees, cos, sin, atan2
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data
import cv2
import time
import tensorflow as tf
from train_ppo.trainer_helper import run_dijkstra
from transforms3d import quaternions

CALC_OBSTACLE_PENALTY = 0

tracking_camera = {
    'yaw': 110,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

class HuskyNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        # assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x,y,z = self.robot.get_position()
        r,p,ya = self.robot.get_rpy()
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x,y,z), (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r,p,ya), (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #contact_ids = set([x[2] for x in f.contact_list()])
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())
        electricity_cost  += self.stall_torque_cost * float(np.square(a).mean())


        steering_cost = self.robot.steering_cost(a)
        debugmode = 0
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = []
        
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        debugmode = 0
        if debugmode:
            print("Husky wall contact:", len(wall_contact))
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_target = 0

        if self.robot.dist_to_target() < 2:
            close_to_target = 0.5

        angle_cost = self.robot.angle_cost()

        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        debugmode = 0
        if debugmode:
            print("angle cost", angle_cost)

        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))
        
        debugmode = 0
        if (debugmode):
            #print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            #print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            print("Obstacle penalty", obstacle_penalty)
            print("Steering cost", steering_cost)
            print("progress", progress)
            #print("electricity_cost")
            #print(electricity_cost)
            #print("joints_at_limit_cost")
            #print(joints_at_limit_cost)
            #print("feet_collision_cost")
            #print(feet_collision_cost)

        rewards = [
            #alive,
            progress,
            wall_collision_cost,
            close_to_target,
            steering_cost,
            #angle_cost,
            #obstacle_penalty
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch)) > 0
        #alive = len(self.robot.parts['top_bumper_link'].contact_list()) == 0

        done = not alive or self.nframe > 250 or height < 0
        #if done:
        #    print("Episode reset")
        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[target_pos[0], target_pos[1], 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs

    ## openai-gym v0.10.5 compatibility
    step = CameraRobotEnv._step


class Husky1DNavigateEnv(HuskyNavigateEnv):
    """HuskyEnv for 1D navigation tasks
    """

    # define constants
    FORWARD = 0
    BACKWARD = 1
    RIGHT = 2
    LEFT = 3
    IDLE = 4

    def __init__(self, config, ob_space_range, gpu_idx=0, action_time=20):
        """
        :param ob_space_range: ob_space_range=(0, 40) has lower bound 0 and upper bound 40
        :param action_time: the number of timesteps each action lasts
        """

        assert action_time > 0
        self.action_time = action_time

        self.ob_low, self.ob_high = ob_space_range
        assert(self.ob_low < self.ob_high and type(self.ob_low) is float and type(self.ob_high) is float)

        HuskyNavigateEnv.__init__(self, config, gpu_idx)

        assert self.config['run-mode'] == 'simulation' \
               or self.config['run-mode'] == 'experiment', \
            'run-mode must be either simulation or experiment!'

        self._is_simulation = self.config['run-mode'] == 'simulation'
        self._require_camera_input = self.config['run-mode'] == 'experiment'
        self._nonviz = self.config['output'] == ['nonviz_sensor']

        # overwrite space dimensions
        self.sensor_space = spaces.Box(low=self.ob_low, high=self.ob_high, shape=(1,))  # 1 dimensional continuous
        self.action_space = spaces.Discrete(2)  # 1 dimensional discrete with 2 elements: 0, 1

        # initialize indicator files
        self.forward_indicator_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   '../..', 'bci-robot-learning', 'objects', 'green_cube.urdf')
        self.backward_indicator_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    '../..', 'bci-robot-learning', 'objects', 'red_cube.urdf')

        self.blue_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '../..', 'bci-robot-learning', 'objects', 'blue_cube.urdf')
        self.black_wall_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '../..', 'bci-robot-learning', 'objects', 'black_wall.urdf')

        # initialize some variables
        self.target_x, self.target_y, self.target_z = self.robot.get_target_position()
        # self.initial_x, self.initial_y, self.initial_z = self.robot.get_position()
        # self.initial_yaw = self.get_robot_yaw()
        p.loadURDF(fileName=self.blue_cube_file,
                   basePosition=[self.target_x, self.target_y, self.target_z],
                   useFixedBase=True)

        p.loadURDF(fileName=self.black_wall_file,
                   basePosition=[self.target_x, self.ob_low, self.target_z],
                   useFixedBase=True)
        p.loadURDF(fileName=self.black_wall_file,
                   basePosition=[self.target_x, self.ob_high, self.target_z],
                   useFixedBase=True)
        self.initial_x, self.initial_y, self.initial_z = self.config['initial_pos']
        _, _, self.initial_yaw = self.config['initial_orn']

        # reset field of view
        self._reset_fov()

    _step = CameraRobotEnv._step
    reset = CameraRobotEnv._reset

    @staticmethod
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

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return [qx, qy, qz, qw]

    @staticmethod
    def convert_angle(angle):
        """
        convert angle to range (0, 2*pi)
        :param angle:
        :return: converted angle
        """
        if angle > 2 * pi:
            angle -= 2 * pi
        elif angle < 0:
            angle += 2 * pi
        return angle

    def _reset_fov(self):
        _, _, _, _, _, _, _, _, cyaw, cpitch, cdist, ctarget = p.getDebugVisualizerCamera()
        yaw_degree = degrees(self.initial_yaw)
        p.resetDebugVisualizerCamera(cameraDistance=cdist,
                                     cameraYaw=yaw_degree - 90,
                                     cameraPitch=cpitch,
                                     cameraTargetPosition=ctarget)

    def get_robot_yaw(self):
        return self.convert_angle(self.quaternion_to_euler(self.robot.get_orientation()))

    def render_observations(self, pose):
        """
        :param pose: pos [x, y, z], quat [w, x, y, z]
        :return: the y coordinate of robot
        """

        if self._require_camera_input:
            self.r_camera_rgb.setNewPose(pose)
            all_dist, all_pos = self.r_camera_rgb.getAllPoseDist(pose)
            top_k = self.find_best_k_views(pose[0], all_dist, all_pos, avoid_block=False)
            # with Profiler("Render to screen"):
            self.render_rgb_filled, self.render_depth, self.render_semantics, self.render_normal, self.render_rgb_prefilled = self.r_camera_rgb.renderOffScreen(
                pose, top_k, rgb=self._require_rgb)

        self.observations = {}
        for output in self.config["output"]:
            try:
                if output == 'nonviz_sensor':
                    pos, orn = pose
                    x, y, z = pos
                    self.render_nonviz_sensor = np.array([y])
                self.observations[output] = getattr(self, "render_" + output)
            except Exception as e:
                raise Exception("Output component {} is not available".format(output))

        # visuals = np.concatenate(visuals, 2)
        return self.observations

    def goal_reached(self, y, tolerance=0.5):
        return abs(y - self.target_y) <= tolerance

    def brake(self, time=100, adjustment_precision=0.0001):
        """
        Brakes
        :param adjustment_precision: the amount of adjustment the robot accounts for its drifting
        """
        print('Braking for {} timesteps'.format(time))
        y_pos = self.robot.get_position()[1]
        for i in range(time):
            self.ensure_orientation()
            new_y_pos = self.robot.get_position()[1]
            if new_y_pos < y_pos:
                self.robot.move_backward(adjustment_precision)
            elif new_y_pos > y_pos:
                self.robot.move_forward(adjustment_precision)
            y_pos = new_y_pos
            self._step(self.IDLE)

    def ensure_orientation(self, tolerance=0.01):
        """
        Ensures the robot's orientation does not change
        """
        yaw = self.get_robot_yaw()
        # print('initial_yaw: {}, yaw: {}'.format(self.initial_yaw, yaw))
        while abs(yaw - self.initial_yaw) > tolerance:
            # print('initial_yaw: {}, yaw: {}'.format(self.initial_yaw, yaw))
            if yaw < self.initial_yaw:
                self._step(self.LEFT)
            else:
                self._step(self.RIGHT)
            yaw = self.get_robot_yaw()

    def step(self, action):
        assert action == self.FORWARD or action == self.BACKWARD

        action_time = self.action_time
        if action == self.FORWARD:
            phrase = 'Accelerating forward'
            indicator_file = self.forward_indicator_file
        else:
            phrase = 'Accelerating backward'
            indicator_file = self.backward_indicator_file

        print("{} for {} timesteps".format(phrase, action_time))
        pos = self.robot.get_position()
        # obs = self.render_observations([self.robot.get_position(), self.robot.get_orientation()])
        rew = -1
        env_done = False
        info = None

        # load initial indicator object
        x0, y0, z0 = pos
        id = p.loadURDF(fileName=indicator_file,
                        basePosition=[x0, y0, z0 + 1],
                        useFixedBase=True)

        # move robot while generating indicator object
        for i in range(action_time):

            # move robot
            self.ensure_orientation()

            # step_start = time.time()
            _, _, _, info = self._step(action, need_return=True)
            # print('robot position xyz : {}'.format(self.robot.get_position()))
            # step_end = time.time()
            # print("The step takes {} s".format(step_end - step_start))

            # get robot odometry
            pos = self.robot.get_position()
            orn = self.robot.get_orientation()
            x, y, z = pos

            # check if robot has reached goal
            if self.goal_reached(y):
                # self.brake(time=100)

                # generate camera input at last step if simulation mode is on
                if self._is_simulation and not self._nonviz:
                    self._require_camera_input = True
                # obs = self.render_observations([self.robot.get_position(), self.robot.get_orientation()])
                obs, _, _, info = self._step(self.IDLE, need_return=True)
                if self._is_simulation and not self._nonviz:
                    self._require_camera_input = False

                rew = 0
                env_done = True
                if indicator_file is not None:
                    p.removeBody(id)

                return obs, rew, env_done, info

            # generate indicator object at the begining
            if i == 0 and indicator_file is not None:
                p.resetBasePositionAndOrientation(bodyUniqueId=id, posObj=[x, y, z + 1], ornObj=orn)

            if i == 3 and indicator_file is not None:
                p.removeBody(id)

        # generate camera input at last step if simulation mode is on
        if self._is_simulation and not self._nonviz:
            self._require_camera_input = True
        # obs = self.render_observations([self.robot.get_position(), self.robot.get_orientation()])
        obs, _, _, info = self._step(self.IDLE, need_return=True)
        if self._is_simulation and not self._nonviz:
            self._require_camera_input = False

        return obs, rew, env_done, info


class Husky2DNavigateEnv(HuskyNavigateEnv):
    """HuskyEnv for 2  D navigation tasks"""

    def __init__(self, config, gpu_idx=0, pos_interval=0.3, orn_interval=30, use_other_room=False):
        """
        :param ob_space_range: ob_space_range=(0, 40) has lower bound 0 and upper bound 40
        :param action_time: the number of timesteps each action lasts
        :param pos_interval: interval for discrete translational displacement
        :param orn_interval: interval for discrete rotational displacement

        * Independent bound for each dimension::
            >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
            Box(2,)
        """

        # assert action_time > 0, 'action_time needs to be greater than 0.'
        # self.action_time = action_time
        assert pos_interval > 0, 'translational displacement interval must be greater than 0.'
        assert orn_interval > 0, 'rotational displacement interval must be greater than 0.'

        # initialize parameters
        self._init_interval_params(pos_interval=pos_interval, orn_interval=orn_interval)
        self._init_tolerance_params()

        HuskyNavigateEnv.__init__(self, config, gpu_idx)

        self._init_configs()
        self._init_constants()
        self._init_spaces()
        self._init_indicators()
        self.config['output'].append('obs')

        self._init_reset_params()

        # reset field of view
        self.reset()
        # self._reset_fov(initial_reset=True)
        self._init_dijkstra(use_other_room)

        if not self._is_simulation:
            import pylsl
            info = pylsl.StreamInfo('Gibson', 'event', 2, 0.0, pylsl.cf_int32, 'gibson')
            self.event_sender = pylsl.StreamOutlet(info)
            self.action_idx = 0

    _step = CameraRobotEnv._step

    '''
    * some functions for __init__()
    '''

    def _init_interval_params(self, pos_interval, orn_interval):

        # parameters for discrete actions
        self._pos_interval = pos_interval
        self._orn_interval = orn_interval

    def _init_tolerance_params(self):

        # tolerance parameters
        self._pos_tolerance = 0.02
        self._orn_tolerance = pi / 180 * 5
        self._ensure_orn_tolerance = 0.03
        self._brake_adjustment_precision = 0.0001

    def _init_configs(self):
        # initialize configurations
        assert self.config['run-mode'] == 'simulation' \
               or self.config['run-mode'] == 'experiment', \
            'run-mode must be either simulation or experiment!'

        self._goal_range = self.config['goal_range']
        self._is_simulation = self.config['run-mode'] == 'simulation'
        self._experiment_step_time = self.config['experiment_step_time']
        # self._require_camera_input = self.config['run-mode'] == 'experiment'
        self._nonviz = self.config['output'] == ['nonviz_sensor']
        self._exclude_backward = self.config['exclude_backward']
        self._use_embedding = self.config['use_embedding']
        self._use_reset = self.config['use_reset']
        self._do_reset_fov = self.config['reset_fov']
        self._do_draw_path = self.config['draw_path']
        self._do_draw_blue_line = self.config['draw_blue_line']
        self._do_draw_green_arrow = self.config['draw_green_arrow']
        self._do_draw_countdown = self.config['show_arrow_countdown']

        self.c_distance = self.config['c_distance']
        self.c_angle = self.config['c_angle']
        self.c_speed = self.config['c_speed']
        self.c_angular_speed = self.config['c_angular_speed']

        # initialize ray casting
        self._use_raycast = self.config['use_raycast']
        if self._use_raycast:
            self._ray_length = self.config['ray_length']
            self._show_rays = self.config['show_rays']
            self._raycast_num = self.config['raycast_num']
            self._raycast_range = self.config['raycast_range'] * pi/6
            self.config['output'].append('ray_distances')
            self._use_goal_info = self.config['use_goal_info']
            self._use_coords_and_orn = self.config['use_coordinate_and_orientation']
            self._ray_height = 0.2
            if self._use_goal_info:
                self.config['output'].append('goal_info')

        assert not (self._use_embedding and self._use_raycast), \
            'cannot use ray casting and embedding at the same time!'

        assert self._exclude_backward, 'backward action has not been fully implemented!'

        if self._is_simulation:
            assert not self._do_draw_green_arrow, 'cannot draw green arrow indicator in simulation mode!!'
            assert not self._do_draw_countdown, 'cannot draw arrow countdown indicator in simulation mode!!'
        else:
            self.config['rl_mode'] = False

        self._use_sensor = not(self._use_embedding or self._use_raycast)

        # number of steps where robot's position and orientation don't change to trigger collision detection
        self._collision_step_limit = self.config['collision_step_limit']
        assert self._collision_step_limit >= 1 and type(self._collision_step_limit) == int, \
            'collision_step_limit must be an integer greater or equal to 1'

        # initialize auto_encoder
        if self._use_embedding:
            self._autoencoder_batch_size = self.config['autoencoder_batch_size']
            self._init_autoencoder()

        if self.config['rl_mode']:
            assert self._is_simulation, "must run rl_mode in simulation!"
            self._use_reset = True
            self._do_draw_path = False
            self._do_draw_green_arrow = False
            self._do_draw_blue_line = False
            self._do_reset_fov = False
            if self._use_raycast:
                self._show_rays = False

        if not self._is_simulation:
            self._use_reset = True
            self._do_reset_fov = True
            self._do_draw_countdown = True
            self._do_draw_green_arrow = self.config["offline"]
            self._show_rays = False

    def _init_constants(self):
        if self._exclude_backward:
            self.FORWARD = 0
            self.RIGHT = 1
            self.LEFT = 2
            self.BACKWARD = 3
        else:
            self.FORWARD = 0
            self.BACKWARD = 1
            self.RIGHT = 2
            self.LEFT = 3
        self.IDLE = 4

        # initialize target and start info
        self.target_position = self.robot.get_target_position()
        self.target_x, self.target_y, self.target_z = self.robot.get_target_position()
        # self.initial_x, self.initial_y, self.initial_z = self.robot.get_position()
        # self.initial_yaw = self.get_robot_yaw()

        self.initial_pos = self.config['initial_pos']
        self.initial_x, self.initial_y, self.initial_z = self.initial_pos
        self.initial_orn = self.config['initial_orn']
        _, _, self.initial_yaw = self.initial_orn

        self.x_range = [-16.3, -4.6]
        self.y_range = [16.0, 27.2]

        # time limit of an episode
        self.episode_len = 0
        self.episode_max_len = 120

    def _init_autoencoder(self):
        checkpoint_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                                            'bci-robot-learning', 'auto_encoder_model')
        self.sess = tf.get_default_session()
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_directory, 'model.ckpt.meta'))  # load graph
        saver.restore(self.sess, os.path.join(checkpoint_directory, 'model.ckpt'))

    def _init_reset_params(self):

        # parameter for resetting the robot
        self._reset_goal_memory()
        self._reached_goal = False
        self._num_steps_stuck = 0
        self._reset_positions = [[self.initial_pos, self.euler_to_quaternion(self.initial_orn)],
                                 [[self.initial_x - 0.9, self.initial_y + 0.9, self.initial_z],
                                  self.euler_to_quaternion([0, 0, pi])],
                                 [[self.initial_x - 1.8, self.initial_y + 0.9, self.initial_z],
                                  self.euler_to_quaternion([0, 0, pi])]]
        # self._reset_positions = [[self.initial_pos, self.euler_to_quaternion(self.initial_orn)],
        #                          [[self.initial_x - 1.5, self.initial_y + 1.5, self.initial_z],
        #                           self.euler_to_quaternion([0, 0, pi])],
        #                          [[self.initial_x - 3, self.initial_y + 0.6, self.initial_z],
        #                           self.euler_to_quaternion([0, 0, pi])]]

    def _init_spaces(self):
        """
        initialize action, observation, sensor spaces
        """

        # overwrite space dimensions
        oo = math.inf
        self.sensor_space = spaces.Box(high=np.array([oo, oo, 0]),
                                       low=np.array([-oo, -oo, 2 * pi]))
        if self._use_embedding:
            self.observation_space = spaces.Box(low=-oo, high=oo, shape=128)
        elif self._use_raycast:
            shape = self._raycast_num
            shape += 3 if self._use_coords_and_orn else 0
            shape += 3 if self._use_goal_info else 0
            self.observation_space = spaces.Box(low=-oo, high=oo, shape=shape)
        else:
            self.observation_space = self.sensor_space

        # discrete with 4 elements: 0, 1, 2, 3
        self.action_space = spaces.Discrete(3) if self._exclude_backward else spaces.Discrete(4)

    def _init_indicators(self):

        # initialize indicator files
        self._blue_cube_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '../..', 'bci-robot-learning', 'objects', 'blue_cube_2D_goal.urdf')
        self._blue_cube_big_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../..', 'bci-robot-learning', 'objects', 'blue_cube_2D_goal_big.urdf')
        self._long_black_wall_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  '../..', 'bci-robot-learning', 'objects', 'long_black_wall.urdf')
        self._long_horiz_wall_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  '../..', 'bci-robot-learning', 'objects', 'long_horizontal_wall.urdf')
        self._short_black_wall_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   '../..', 'bci-robot-learning', 'objects', 'short_black_wall.urdf')
        self._long_obstacle_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../..', 'bci-robot-learning', 'objects', 'long_obstacle.urdf')
        self._short_obstacle_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 '../..', 'bci-robot-learning', 'objects', 'short_obstacle.urdf')
        self._green_arrow_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              '../..', 'bci-robot-learning', 'objects', 'green_arrow.urdf')
        self._white_arrow_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              '../..', 'bci-robot-learning', 'objects', 'white_arrow.urdf')
        self._red_arrow_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '../..', 'bci-robot-learning', 'objects', 'red_arrow.urdf')

        # load obstacle and goal indicators
        if self._goal_range >= 0.75:
            p.loadURDF(fileName=self._blue_cube_big_file,
                       basePosition=[self.target_x, self.target_y, self.target_z],
                       useFixedBase=True)
        else:
            p.loadURDF(fileName=self._blue_cube_file,
                       basePosition=[self.target_x, self.target_y, self.target_z],
                       useFixedBase=True)

        self.indicator_id = None
        self.indicator_id2 = None
        self.countdown_target_id = None
        self.countdown_pointer_id = None
        self.correct_actions = None
        self.indicator_height = 0.5
        self.countdown_height = 0.8
        self.indicator_color = {'blue': (0, 0, 240)}
        self.path_colors = {'red': (240, 0, 0),
                            'yellow': (240, 240, 0),
                            'black': (0, 0, 0),
                            'purple': (128, 0, 128),
                            'white': (255, 250, 250)
                            }
        self.ray_color = {'sky_blue': (0, 240, 240)}
        self._path_colors_available = set(list(self.path_colors.keys()))

        long_wall_locs = [[-4.0, 22.0, 0.15], [-17.0, 20.6, 0.15], [-23.4, 48.4, 0.08], [-14, 48.8, 0.164]]
        short_wall_locs = [[-14.0, 27.7, 0.177], [-14.4, 15.2, 0.138], [-4, 23.7, 0.155], [-4.5, 18.5, 0.172]]
        long_horiz_locs = [[-19.1, 51.0, 0.1], [-19.1, 47.0, 0.1]]
        long_obstacle_loc = [-10.09, 18.2, 0.145]
        short_obstacle_loc = [-10.11, 25.0, 0.129]

        for loc in long_wall_locs:
            p.loadURDF(fileName=self._long_black_wall_file,
                       basePosition=loc,
                       useFixedBase=True)

        for loc in long_horiz_locs:
            p.loadURDF(fileName=self._long_horiz_wall_file,
                       basePosition=loc,
                       useFixedBase=True)

        for loc in short_wall_locs:
            p.loadURDF(fileName=self._short_black_wall_file,
                       basePosition=loc,
                       useFixedBase=True)

        p.loadURDF(fileName=self._long_obstacle_file,
                   basePosition=long_obstacle_loc,
                   useFixedBase=True)

        p.loadURDF(fileName=self._short_obstacle_file,
                   basePosition=short_obstacle_loc,
                   useFixedBase=True)

        self.bad_move_count = 0
        self.good_move_count = 0
        self.total_move_count = 0

    def _init_dijkstra(self, use_other_room=False):

        # initialize dijkstra function
        self.judge_action, self.get_good_actions, self.generate_shortest_paths, \
        self.get_neighbor_from_action, self.idx_2_cor, self.check_collision = run_dijkstra(self, self.target_position, use_other_room=use_other_room)

    '''
    * some static helper functions
    '''

    @staticmethod
    def quaternion_to_euler(orn, yaw_only=True):
        """
        :param orn: a 4-element list of quarternion angles
        :return: a 3-element list of eulerian angles [roll, pitch, yaw]
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
            return [roll, pitch, yaw]

    @staticmethod
    def euler_to_quaternion(rpy):
        roll, pitch, yaw = rpy
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return [qx, qy, qz, qw]

    @staticmethod
    def convert_angle(angle):
        """
        convert angle to range (0, 2*pi)
        :param angle:
        :return: converted angle
        """
        if angle > 2 * pi:
            angle -= 2 * pi
        elif angle < 0:
            angle += 2 * pi
        return angle

    @staticmethod
    def calc_2d_dist(p, q):
        """
        :param p: robot position list p
        :param q: robot position list q
        :return: the 2D Euclidean distance between p and q
        """
        p = [p[0], p[1]]
        q = [q[0], q[1]]
        return sum((p - q) ** 2 for p, q in zip(p, q)) ** 0.5

    def reset_goal_range(self, goal_range=0.5):
        if self._goal_range >= 0.75 and goal_range >= 0.75:
            return
        if self._goal_range < 0.75 and goal_range < 0.75:
            return
        if self._goal_range > goal_range:
            return

        self._goal_range = goal_range

        # load goal indicators
        if self._goal_range >= 0.75:
            p.loadURDF(fileName=self._blue_cube_big_file,
                       basePosition=[self.target_x, self.target_y, self.target_z],
                       useFixedBase=True)
        else:
            p.loadURDF(fileName=self._blue_cube_file,
                       basePosition=[self.target_x, self.target_y, self.target_z],
                       useFixedBase=True)

    def reset_state_space(self,
                          use_embedding=False,
                          use_raycast=True,
                          use_coords_and_orn=True,
                          use_goal_info=True,
                          raycast_num=10,
                          raycast_range=6,
                          start_varying_range=0.2):
        """
        reset observation spaces
        """

        # overwrite space dimensions
        oo = math.inf
        self._use_embedding = use_embedding
        self._use_raycast = use_raycast
        self._use_coords_and_orn = use_coords_and_orn
        self._use_goal_info = use_goal_info
        self._raycast_num = raycast_num
        self._raycast_range = raycast_range * pi/6

        assert not (self._use_embedding and self._use_raycast), 'cannot use embedding and raycast simultaneously!'

        if self._use_embedding:
            self.observation_space = spaces.Box(low=-oo, high=oo, shape=128)
        elif self._use_raycast:
            shape = self._raycast_num
            shape += 3 if self._use_coords_and_orn else 0
            shape += 3 if self._use_goal_info else 0
            self.observation_space = spaces.Box(low=-oo, high=oo, shape=shape)
        else:
            self.observation_space = self.sensor_space

        self.config["random"]["random_init_x_range"] = [-start_varying_range, start_varying_range]
        self.config["random"]["random_init_y_range"] = [-start_varying_range, start_varying_range]

    def generate_embeddings(self, x):
        """
        :param x: array of size (1, 256, 256, 4)
        :return embedding: array of size (1, 4, 4, 256)
        :return loss: float
        :return output: reconstructed image, array of (1, 256, 256, 4)
        """
        x = np.repeat(x, self._autoencoder_batch_size, axis=0)
        embedding, loss, output = self.sess.run(
            ['autoencoder/embeddings:0', 'mean_squared_error/value:0', 'autoencoder/out:0'],
            feed_dict={'IteratorGetNext:0': x,
                       'IteratorGetNext:1': x,
                       'PlaceholderWithDefault_2:0': False})
        return embedding[[0]], loss, output[[0]]

    def _reset_fov(self, initial_reset=False):

        if not self._do_reset_fov:
            return

        _, _, _, _, _, _, _, _, cyaw, cpitch, cdist, ctarget = p.getDebugVisualizerCamera()
        if initial_reset:
            yaw_degree = degrees(self.initial_yaw)
        else:
            yaw_degree = degrees(self.get_robot_yaw())
        p.resetDebugVisualizerCamera(cameraDistance=cdist,
                                     cameraYaw=yaw_degree - 90,
                                     cameraPitch=cpitch,
                                     cameraTargetPosition=ctarget)

    def _reset_goal_memory(self):
        self._prev_dist_to_goal = self.dist_to_target()
        self._prev_angle_from_goal = np.abs(self.angle_from_target())

    def random_reset(self):
        self.episode_len = 0
        CameraRobotEnv._reset(self)

        if self.config["random"]["random_initial_pose"]:

            pos = self.robot.robot_body.get_position()
            orn = self.robot.robot_body.get_orientation()

            x_range = self.config["random"]["random_init_x_range"]
            y_range = self.config["random"]["random_init_y_range"]
            z_range = self.config["random"]["random_init_z_range"]
            # r_range = self.config["random"]["random_init_rot_range"]

            while True:
                new_pos = [pos[0] + self.np_random.uniform(low=x_range[0], high=x_range[1]),
                           pos[1] + self.np_random.uniform(low=y_range[0], high=y_range[1]),
                           pos[2] + self.np_random.uniform(low=z_range[0], high=z_range[1])]
                new_orn = orn
                x, y, _ = new_pos
                yaw = self.convert_angle(self.quaternion_to_euler(new_orn))
                state = (x, y, yaw)
                if not self.check_collision(state):
                    break
        else:
            pose = random.choice(self._reset_positions)
            new_pos, new_orn = pose
        # print('pos: {}, orn: {}'.format(pos, orn))
        self.robot.reset_new_pose(new_pos, new_orn)
        return self.render_observations([new_pos, new_orn])

    def in_collision(self):
        wall_contact = []
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        return len(wall_contact) > 0

    def dist_to_target(self):
        x, y, _ = self.get_robot_xyz()
        return ((x - self.target_x)**2 + (y - self.target_y)**2)**(1/2)

    def angle_from_target(self):
        x, y, _ = self.get_robot_xyz()
        dy = self.target_y - y
        dx = self.target_x - x
        goal_angle = self.convert_angle(atan2(dy, dx))
        yaw = self.get_robot_yaw()
        relative_angle = goal_angle - yaw
        if relative_angle > pi:
            relative_angle -= 2 * pi
        if relative_angle < -pi:
            relative_angle += 2 * pi
        return relative_angle

    def get_robot_yaw(self):
        return self.convert_angle(self.quaternion_to_euler(self.robot.get_orientation()))

    def get_robot_xyz(self):
        return self.robot.get_position()

    def get_robot_state(self):
        x, y, z = self.robot.get_position()
        yaw = self.get_robot_yaw()
        return x, y, yaw

    def _get_indicator_orn(self, action):
        """
        :return: given the current robot orn in quaternion, return the indicator orn in quaternion
        """

        if action == self.FORWARD:
            indicator_yaw_offset = 0
        elif action == self.LEFT:
            indicator_yaw_offset = pi/2
        elif action == self.RIGHT:
            indicator_yaw_offset = -pi/2
        else:
            indicator_yaw_offset = pi

        orn = self.robot.get_orientation()
        robot_rpy = self.quaternion_to_euler(orn, yaw_only=False)
        roll, pitch, yaw = robot_rpy
        indicator_rpy = roll, pitch, yaw + indicator_yaw_offset
        indicator_quat = self.euler_to_quaternion(indicator_rpy)
        return indicator_quat

    def _get_ray_from_and_to(self):
        x, y, z = self.get_robot_xyz()
        robot_yaw = self.get_robot_yaw()
        ray_from = [[x, y, z + self._ray_height] for _ in range(self._raycast_num)]
        ray_angles = [robot_yaw - self._raycast_range/2 + multiple * self._raycast_range/(self._raycast_num - 1)
                      for multiple in range(self._raycast_num)] if self._raycast_num != 1 else [robot_yaw]
        ray_to = []
        for ray_angle in ray_angles:
            ray_to_position = [x + self._ray_length * cos(ray_angle),
                               y + self._ray_length * sin(ray_angle),
                               z + self._ray_height]
            ray_to.append(ray_to_position)

        return ray_from, ray_to

    def _get_ray_hit_dists(self):
        ray_from, ray_to = self._get_ray_from_and_to()
        # print('from: {}, to: {}'.format(ray_from, ray_to))
        results = p.rayTestBatch(ray_from, ray_to, self.robot_tracking_id)
        hit_dists = []
        for i in range(self._raycast_num):
            hit_object_id = results[i][0]

            if hit_object_id < 0:
                ray_end = ray_to[i]
                hit_dist = self._ray_length
            else:
                ray_end = results[i][3]
                hit_dist = self.calc_2d_dist(ray_from[i], ray_end)

            hit_dists.append(hit_dist) # normalization

            # draw rays
            if self._show_rays:
                x1, y1, z1 = ray_from[i]
                ray_indicator_from = [x1, y1, z1]
                x2, y2, z2 = ray_end
                ray_indicator_to = [x2, y2, z2]
                p.addUserDebugLine(ray_indicator_from,
                                   ray_indicator_to,
                                   self.ray_color['sky_blue'],
                                   lineWidth=1000)
        # print('hit: {}'.format(hit_dists))

        return hit_dists

    def _x_normalizer(self, x):
        low, high = self.x_range
        return (x - low)/(high - low)

    def _y_normalizer(self, y):
        low, high = self.y_range
        return (y - low)/(high - low)

    def _dist_normalizer(self, dist):
        return dist/self._ray_length

    '''
    * override function in super class
    '''
    def render_observations(self, pose):
        """
        :param pose: pos [x, y, z], quat [w, x, y, z]
        :return: the y coordinate of robot
        """

        if self._require_camera_input:
            self.r_camera_rgb.setNewPose(pose)
            all_dist, all_pos = self.r_camera_rgb.getAllPoseDist(pose)
            top_k = self.find_best_k_views(pose[0], all_dist, all_pos, avoid_block=False)
            # with Profiler("Render to screen"):
            self.render_rgb_filled, self.render_depth, self.render_semantics, \
            self.render_normal, self.render_rgb_prefilled = self.r_camera_rgb.renderOffScreen(
                pose, top_k, rgb=self._require_rgb)

        self.observations = {}

        for output in self.config["output"]:
            try:
                if output == 'nonviz_sensor':

                    # output coordinate and orienation
                    x, y, _ = self.get_robot_xyz()
                    yaw = self.get_robot_yaw()
                    self.render_nonviz_sensor = np.array([x, y, yaw])

                if output == 'ray_distances':
                    dists = self._get_ray_hit_dists()
                    self.render_ray_distances = np.array(dists)

                if output == 'goal_info':
                    dist_to_goal = self.dist_to_target()
                    x, y, _ = self.get_robot_xyz()
                    dy = self.target_y - y
                    dx = self.target_x - x
                    angle_from_goal = atan2(dy, dx)
                    yaw = self.get_robot_yaw()
                    self.render_goal_info = np.array([dist_to_goal, angle_from_goal, yaw])
                    # angle_relative_to_goal = self.angle_from_target()
                    # self.render_goal_info = np.array([dx, dy, angle_relative_to_goal])

                if output == 'obs':
                    if self._use_raycast:
                        dists = self.render_ray_distances
                        normalized_ray_distances = self._dist_normalizer(dists)
                        obs = normalized_ray_distances
                        if self._use_coords_and_orn:
                            x, y, yaw = self.render_nonviz_sensor
                            normalized_nonviz = np.array([self._x_normalizer(x), self._y_normalizer(y), yaw])
                            obs = np.concatenate((obs, normalized_nonviz))
                        if self._use_goal_info:
                            dist, angle_from_goal, yaw = self.render_goal_info
                            normalized_goal_info = np.array([self._dist_normalizer(dist), angle_from_goal, yaw])
                            # dx, dy, angle = self.render_goal_info
                            # normalized_goal_info = np.array([self._dist_normalizer(dx), self._dist_normalizer(dy), angle])
                            obs = np.concatenate((obs, normalized_goal_info))
                        self.render_obs = obs
                    elif self._use_embedding:
                        if self._require_camera_input:
                            # output autoencoded matrix
                            rgbd = np.concatenate([self.render_rgb_filled, self.render_depth], axis=-1)
                            rgbd = np.expand_dims(rgbd, axis=0)
                            embedding, *_ = self.generate_embeddings(rgbd)
                            self.render_obs = embedding.flatten()
                        else:
                            self.render_obs = None
                    else:
                        self.render_obs = self.render_nonviz_sensor

                self.observations[output] = getattr(self, "render_" + output)
            except Exception as e:
                raise Exception("Output component {} is not available".format(output))

        # visuals = np.concatenate(visuals, 2)
        return self.observations

    def reset(self):
        self.episode_len = 0
        obs = HuskyNavigateEnv._reset(self)

        if self.config['run-mode'] == 'experiment':
            self.robot.is_discrete = False
            noop = np.zeros(4)
            HuskyNavigateEnv.step(self, noop)
            self.robot.is_discrete = True
            self.robot.is_discrete = True
            self._reset_fov(initial_reset=True)

        self._reset_goal_memory()
        return obs

    def goal_reached(self):
        if abs(self.dist_to_target() <= self._goal_range):
            self._reached_goal = True
            return True
        else:
            return False

    def _get_indicator_line_endpoint(self, action):
        x, y, z = self.get_robot_xyz()
        yaw = self.get_robot_yaw()
        theta = yaw
        if action == self.FORWARD:
            theta += 0
        elif action == self.BACKWARD:
            theta += pi
        elif action == self.LEFT:
            theta += pi/3
        else:
            theta -= pi/3
        dist_from_self = 5 * self._pos_interval
        end_x, end_y, end_z = x + dist_from_self * cos(theta), y + dist_from_self * sin(theta), z-0.1
        return [end_x, end_y, end_z]

    def _get_indicator_line_startpoint(self, action, dist_from_self=0):
        """
        :param dist_from_husky: how far away the starting point of indicator is away from husky
        :return: the starting point of the path indicator line
        """
        x, y, z = self.get_robot_xyz()
        yaw = self.get_robot_yaw()
        theta = yaw
        if action == self.FORWARD:
            theta += 0
        elif action == self.BACKWARD:
            theta += pi
        elif action == self.LEFT:
            theta += pi/3
        else:
            theta -= pi/3
        start_x, start_y, start_z = x + dist_from_self * cos(theta), y + dist_from_self * sin(theta), z-0.1
        return [start_x, start_y, start_z]

    def draw_indicator_line(self, action):
        start_xyz = self._get_indicator_line_startpoint(action=action)
        end_xyz = self._get_indicator_line_endpoint(action=action)
        # print('drawing indicator line for action: {}, start: {}, end: {}'.format(action, start_xyz, end_xyz))
        # self.remove_indicator_line()
        id = p.addUserDebugLine(lineFromXYZ=start_xyz, lineToXYZ=end_xyz,
                                lineWidth=1000, lineColorRGB=self.indicator_color['blue'])

        return id

    def _get_other_action(self, action):
        correct_action = action + 1
        if correct_action > 2:
            correct_action = 0
        return correct_action

    def _draw_green_arrow(self):

        assert self.correct_actions is not None and self.correct_actions, \
            'must get correct actions or fix no correct action returned error before drawing green arrow!'

        assert len(self.correct_actions) == 1 or len(self.correct_actions) == 2, \
            'there must be 1 or 2 correct action(s)!'

        indicator_orn = self._get_indicator_orn(action=self.correct_actions[0])

        x, y, z = self.get_robot_xyz()

        # generate indicator object
        if self.indicator_id is None:
            self.indicator_id = p.loadURDF(fileName=self._green_arrow_file,
                                           basePosition=[x, y, z + self.indicator_height],
                                           useFixedBase=True)

        p.resetBasePositionAndOrientation(bodyUniqueId=self.indicator_id,
                                          posObj=[x, y, z + self.indicator_height],
                                          ornObj=indicator_orn)
        if len(self.correct_actions) == 2:

            indicator_orn = self._get_indicator_orn(action=self.correct_actions[1])
            # generate indicator object
            if self.indicator_id2 is None:
                self.indicator_id2 = p.loadURDF(fileName=self._green_arrow_file,
                                                basePosition=[x, y, z + self.indicator_height],
                                                useFixedBase=True)

            p.resetBasePositionAndOrientation(bodyUniqueId=self.indicator_id2,
                                              posObj=[x, y, z + self.indicator_height],
                                              ornObj=indicator_orn)

    def _get_countdown_orns(self):
        orn = self.robot.get_orientation()
        robot_rpy = self.quaternion_to_euler(orn, yaw_only=False)
        roll, pitch, yaw = robot_rpy
        target_rpy = roll + pi/2, pitch, yaw - pi/2
        target_quat = self.euler_to_quaternion(target_rpy)
        pointer_rpy = roll + pi/2, pitch, yaw + pi/2
        pointer_quat = self.euler_to_quaternion(pointer_rpy)
        return target_quat, pointer_quat

    def _get_pointer_orn(self, pitch_progress):
        orn = self.robot.get_orientation()
        robot_rpy = self.quaternion_to_euler(orn, yaw_only=False)
        roll, pitch, yaw = robot_rpy
        pointer_rpy = roll + pi/2, pitch - pitch_progress, yaw + pi/2
        pointer_quat = self.euler_to_quaternion(pointer_rpy)
        return pointer_quat

    def _draw_countdown(self, countdown_time):
        x, y, z = self.get_robot_xyz()
        if self.countdown_target_id is None:
            self.countdown_target_id = p.loadURDF(fileName=self._white_arrow_file,
                                                  basePosition=[x, y, z + self.countdown_height],
                                                  useFixedBase=True)

        if self.countdown_pointer_id is None:
            self.countdown_pointer_id = p.loadURDF(fileName=self._red_arrow_file,
                                                  basePosition=[x, y, z + self.countdown_height],
                                                  useFixedBase=True)

        target_orn, pointer_orn = self._get_countdown_orns()

        p.resetBasePositionAndOrientation(bodyUniqueId=self.countdown_target_id,
                                          posObj=[x, y, z + self.countdown_height],
                                          ornObj=target_orn)

        p.resetBasePositionAndOrientation(bodyUniqueId=self.countdown_pointer_id,
                                          posObj=[x, y, z + self.countdown_height],
                                          ornObj=target_orn)

        start_time = time.time()
        while True:
            current_time = time.time()
            time_elapsed = current_time - start_time
            if time_elapsed > countdown_time:
                break

            pitch_progress = time_elapsed/countdown_time * pi
            pointer_progress_orn = self._get_pointer_orn(pitch_progress)
            p.resetBasePositionAndOrientation(bodyUniqueId=self.countdown_pointer_id,
                                              posObj=[x, y, z + self.countdown_height],
                                              ornObj=pointer_progress_orn)


        # make them disappear
        p.resetBasePositionAndOrientation(bodyUniqueId=self.countdown_pointer_id,
                                          posObj=[0, 0, 0],
                                          ornObj=self.euler_to_quaternion([0, 0, 0]))

        p.resetBasePositionAndOrientation(bodyUniqueId=self.countdown_target_id,
                                          posObj=[0, 0, 0],
                                          ornObj=self.euler_to_quaternion([0, 0, 0]))

    def remove_indicators(self, id=None):
        # print('removing indicator line!')
        if id is None:
            p.removeAllUserDebugItems()
        else:
            p.removeUserDebugItem(itemUniqueId=id)
        self._reset_path_colors_available()
        # if self.indicator_id is not None:
        #     p.resetBasePositionAndOrientation(bodyUniqueId=self.indicator_id,
        #                                       posObj=[0, 0, 0],
        #                                       ornObj=self.euler_to_quaternion([0, 0, 0]))
        if self.indicator_id2 is not None:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.indicator_id2,
                                              posObj=[0, 0, 0],
                                              ornObj=self.euler_to_quaternion([0, 0, 0]))

    def _draw_path_seg(self, start_xyz, end_xyz, path_color):
        path_rgb = self.path_colors[path_color]
        p.addUserDebugLine(lineFromXYZ=start_xyz, lineToXYZ=end_xyz,
                           lineWidth=1000, lineColorRGB=path_rgb)

    def draw_path(self):
        x, y, z = self.get_robot_xyz()
        yaw = self.get_robot_yaw()
        paths = self.generate_shortest_paths([x, y, yaw])
        # print('paths: {}'.format(paths))
        for path in paths:
            path_color = random.choice(tuple(self._path_colors_available))
            self._path_colors_available.remove(path_color)
            prev = path[0]
            for node in path:
                if node == prev:
                    continue

                # unpack and pack information
                prev_x, prev_y, _ = prev
                cur_x, cur_y, _ = node
                start_xyz = prev_x, prev_y, z
                end_xyz = cur_x, cur_y, z

                # draw the path segment
                self._draw_path_seg(start_xyz, end_xyz, path_color)

                prev = node

    def _reset_path_colors_available(self):
        self._path_colors_available = set(list(self.path_colors.keys()))

    def _determine_pause_or_resume(self):

        # pause/resume simulation when space key is pressed
        pressed_keys = self.get_key_pressed()
        # print('keys during check pausing: {}'.format(pressed_keys))
        if ord(' ') in pressed_keys:
            print('Pausing...')
            self.total_move_count = 0
            self.bad_move_count = 0
            self.good_move_count = 0
            pausing = True
            while pausing:
                time.sleep(0.1)
                pressed_keys = self.get_key_pressed()
                # print('keys during check resuming: {}'.format(pressed_keys))
                if ord('r') in pressed_keys:
                    print('Resuming!')
                    return

    def _reset_pose(self, action):
        x, y, z = self.get_robot_xyz()
        roll, pitch, yaw = self.quaternion_to_euler(self.robot.get_orientation(), yaw_only=False)
        (new_x_idx, new_y_idx, new_yaw_idx), met_collision = self.get_neighbor_from_action(x, y, yaw, action)
        x, y, yaw = self.idx_2_cor(new_x_idx, new_y_idx, new_yaw_idx)
        pos = x, y, z
        orn = self.euler_to_quaternion([roll, pitch, yaw])
        self.robot.reset_new_pose(pos, orn)
        # time.sleep(0.2)

    def _get_rich_reward(self):
        dist = self.dist_to_target()
        angle_target = self.angle_from_target()
        angle = np.abs(angle_target)

        reward = self.c_speed * (self._prev_dist_to_goal - dist) + \
                 self.c_angular_speed * (self._prev_angle_from_goal - angle) + \
                 self.c_angle * angle + \
                 self.c_distance * dist

        self._prev_dist_to_goal = dist
        self._prev_angle_from_goal = angle

        x, y, _ = self.get_robot_xyz()
        # print("x {:>5.2f}, y {:>5.2f}, yaw {:>5.3f}, rew {:>5.3f}".format(x, y, yaw, reward))

        return reward

    def _rotate(self, action):
        if self._use_reset:
            self._reset_pose(action)
            if self._is_simulation:
                self._reset_fov()
            else:
                if self._do_draw_green_arrow:
                    self._draw_green_arrow()
            self.goal_reached()
            return

        rotate_start = time.time()

        self.robot.is_discrete = False
        timeout = 300
        base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])

        orn = self.robot.get_orientation()
        # indicator_orn = self._get_indicator_orn(orn, pi / 2 if action == self.LEFT else -pi / 2)

        x, y, z = self.get_robot_xyz()
        yaw = self.get_robot_yaw()
        if action == self.LEFT:
            yaw_goal = yaw + pi/180 * self._orn_interval
        else:
            yaw_goal = yaw - pi/180 * self._orn_interval
        if yaw_goal >= 2 * pi:
            yaw_goal -= 2 * pi
        if yaw_goal < 0:
            yaw_goal += 2 * pi

        error = pi/180 * self._orn_interval
        omega = self.robot_body.angular_speed()[-1]

        kp_omega = 200

        while abs(error) > 2 * pi/180 or np.abs(omega) > 1e-1:

            if self._do_draw_green_arrow:
                self._draw_green_arrow()

            timeout -= 1

            yaw = self.get_robot_yaw()
            omega = self.robot_body.angular_speed()[-1]

            error = yaw_goal - yaw
            if error < -pi:
                error += 2 * pi
            if error > pi:
                error -= 2 * pi
            angular_acc = kp_omega * error - 2 * np.sqrt(kp_omega) * omega
            # print(angular_acc)
            # angular_acc = np.clip(angular_acc, -20, 20)
            HuskyNavigateEnv.step(self, angular_acc * base_action_omage)

            self._reset_fov()

            if self.goal_reached():
                break
            if time.time() - rotate_start > self._experiment_step_time * 0.4:
                # print("error", error / pi * 180, "omega", np.abs(omega), "action", action)
                break
            if self.in_collision():
                break

        self.robot.is_discrete = True

    def _move(self, action):
        if self._use_reset:
            self._reset_pose(action)
            if self._is_simulation:
                self._reset_fov()
            else:
                if self._do_draw_green_arrow:
                    self._draw_green_arrow()
            self.goal_reached()
            return

        move_start = time.time()

        self.robot.is_discrete = False
        base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])
        base_action_v = np.array([0.001, 0.001, 0.001, 0.001])

        x_init, y_init, z = self.get_robot_xyz()
        yaw_init = self.get_robot_yaw()
        v = self.robot.robot_body.speed()[0]

        kp_v = 40
        kp_omega = 20

        error = self._pos_interval
        while np.abs(error) > 0.01 or np.abs(v) > 1e-2:
            if self._do_draw_green_arrow:
                self._draw_green_arrow()

            x, y, z = self.get_robot_xyz()
            yaw = self.get_robot_yaw()
            v = self.robot.robot_body.speed()[0]
            omega = self.robot_body.angular_speed()[-1]

            dist = self.calc_2d_dist((x_init, y_init), (x, y))
            error = (self._pos_interval - dist) if action == self.FORWARD else dist - self._pos_interval
            if (self._pos_interval - dist) * v < 0:
                v = 0
            acc = kp_v * error - 2 * np.sqrt(kp_v) * v
            angular_acc = kp_omega * (yaw_init - yaw) - 2 * np.sqrt(kp_omega) * omega
            acc = np.clip(acc, -20, 20)
            HuskyNavigateEnv.step(self, acc * base_action_v + angular_acc * base_action_omage)

            self._reset_fov()

            if self.goal_reached():
                break
            if time.time() - move_start > self._experiment_step_time * 0.4:
                # print("error", error, "v", np.abs(v))
                break
            if self.in_collision():
                break

        # print(timeout)
        self.robot.is_discrete = True

    def step(self, action):

        # ideal_step_time = np.random.uniform(self._experiment_step_time - 0.2, self._experiment_step_time + 0.2)
        ideal_step_time = self._experiment_step_time
        step_start = time.time()

        # assert check
        assert action == self.FORWARD or action == self.BACKWARD \
               or action == self.LEFT or action == self.RIGHT \
               or action == self.IDLE, ' action must be inside Discrete(4) range or IDLE(4). '

        rewards = {'rich': 0, 'sparse': 0}

        # load initial indicator object
        pos = self.robot.get_position()
        yaw = self.get_robot_yaw()
        x0, y0, z0 = pos

        # draw correct path(s)
        if self._do_draw_path:
            self.draw_path()

        # get good actions
        state = (x0, y0, yaw)
        if action != self.IDLE:
            self.correct_actions = self.get_good_actions(state)

            # # for debugging only
            # if random.random() > 0.5:
            #     self.correct_actions = [1, 2]

            if not self.correct_actions:
                self.correct_actions = [self._get_other_action(action=action)]

        # print('correct actions: {}'.format(self.correct_actions))

        # draw blue line correct action indicator(s)
        if self._do_draw_blue_line:
            for correct_action in self.correct_actions:
                self.draw_indicator_line(action=correct_action)

        # draw green arrow correct action indicator(s)
        if self._do_draw_green_arrow:
            self._reset_fov()
            self._draw_green_arrow()
        if self._do_draw_countdown:
            self._draw_countdown(self._experiment_step_time * 0.4)

        if action != self.IDLE:
            good_move = self.judge_action(state, action)
            if not self._is_simulation:
                self.event_sender.push_sample([good_move, self.action_idx])

        # perform action
        if action == self.FORWARD or action == self.BACKWARD:
            self._move(action)
        elif action == self.LEFT or action == self.RIGHT:
            self._rotate(action)

        if not self._is_simulation and self._use_reset:
            cur_time = time.time()
            while time.time() - cur_time < self._experiment_step_time * 0.4:
                continue
            self._reset_fov()

        # print move count info
        if not self._is_simulation and action != self.IDLE:
            self.total_move_count += 1
            if self.judge_action(state, action):
                self.good_move_count += 1
            else:
                self.bad_move_count += 1
            print('Total Moves: {}, Good moves: {}, Bad moves: {}\n\n'.format(self.total_move_count,
                                                                              self.good_move_count,
                                                                              self.bad_move_count))
        # generate camera input at last step if simulation mode is on
        if self._is_simulation and not self._nonviz:
            self._require_camera_input = True

        obs, _, _, info = self._step(self.IDLE)

        if self._is_simulation and not self._nonviz:
            self._require_camera_input = False

        # check if time out
        self.episode_len += 1
        episode_timeout = self.episode_len >= self.episode_max_len

        # check if the robot gets stuck ()
        pos_new = self.robot.get_position()
        yaw_new = self.get_robot_yaw()
        if self.calc_2d_dist(pos, pos_new) <= self._pos_tolerance and abs(yaw - yaw_new) <= self._orn_tolerance:
            self._num_steps_stuck += 1

        if self._reached_goal:
            print('Reached Goal! Resetting')
            rewards['sparse'] = 100
            rewards['rich'] = 100
            env_done = True
            self._reached_goal = False
        elif self.in_collision() or self._num_steps_stuck >= self._collision_step_limit:
            self._num_steps_stuck = 0
            rewards['sparse'] = -100
            rewards['rich'] = -100
            env_done = True
        else:
            rewards['sparse'] = -1
            rewards['rich'] = self._get_rich_reward()
            env_done = episode_timeout

        step_end = time.time()
        self._determine_pause_or_resume()
        self.correct_actions = None

        if action != self.IDLE:
            self.remove_indicators()

        if not self._is_simulation and step_end - step_start < ideal_step_time:
            sleep_time = ideal_step_time - (step_end - step_start)
            time.sleep(sleep_time)
            time_after_sleep = time.time()
        # print("This step took {} seconds".format(time.time() - step_start))

        # self._reset_fov()

        return obs, rewards, env_done, info


class HuskyNavigateSpeedControlEnv(HuskyNavigateEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        HuskyNavigateEnv.__init__(self, config, gpu_idx)
        self.robot.keys_to_action = {
            (ord('s'), ): [-0.5,0], ## backward
            (ord('w'), ): [0.5,0], ## forward
            (ord('d'), ): [0,-0.5], ## turn right
            (ord('a'), ): [0,0.5], ## turn left
            (): [0,0]
        }

        self.base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])
        self.base_action_v = np.array([0.001, 0.001, 0.001, 0.001])
        self.action_space = gym.spaces.Discrete(5)
        #control_signal = -0.5
        #control_signal_omega = 0.5
        self.v = 0
        self.omega = 0
        self.kp = 100
        self.ki = 0.1
        self.kd = 25
        self.ie = 0
        self.de = 0
        self.olde = 0
        self.ie_omega = 0
        self.de_omega = 0
        self.olde_omage = 0


    def _step(self, action):
        control_signal, control_signal_omega = action
        self.e = control_signal - self.v
        self.de = self.e - self.olde
        self.ie += self.e
        self.olde = self.e
        pid_v = self.kp * self.e + self.ki * self.ie + self.kd * self.de

        self.e_omega = control_signal_omega - self.omega
        self.de_omega = self.e_omega - self.olde_omage
        self.ie_omega += self.e_omega
        pid_omega = self.kp * self.e_omega + self.ki * self.ie_omega + self.kd * self.de_omega

        obs, rew, env_done, info = HuskyNavigateEnv.step(self, pid_v * self.base_action_v + pid_omega * self.base_action_omage)

        self.v = obs["nonviz_sensor"][3]
        self.omega = obs["nonviz_sensor"][-1]

        return obs,rew,env_done,info

    ## openai-gym v0.10.5 compatibility
    step  = _step


class HuskyGibsonFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        print(self.config["envname"])
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"
        
        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100
        
    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def _flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)
        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.get_position()


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 3000 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def _rewards(self, action = None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        if self.flag_timeout > 225:
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("progress")
            print(progress)

        obstacle_penalty = 0

        #print("obs dist %.3f" %self.obstacle_dist)
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        rewards = [
            alive_score,
            progress,
            obstacle_penalty
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > 500
        if (debugmode):
            print("alive=")
            print(alive)
        print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0 or (self.flag_timeout < 225 and self.robot.walk_target_dist < 0.8):
            self._flag_reposition()
        self.flag_timeout -= 1

        if "depth" in self.config["output"]:
            depth_obs = self.get_observations()["depth"]
            x_start = int(self.windowsz/2-16)
            x_end   = int(self.windowsz/2+16)
            y_start = int(self.windowsz/2-16)
            y_end   = int(self.windowsz/2+16)
            self.obstacle_dist = (np.mean(depth_obs[x_start:x_end, y_start:y_end, -1]))

        return state, reward, done, meta

    ## openai-gym v0.10.5 compatibility
    step  = _step


class HuskySemanticNavigateEnv(SemanticRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        self.config = self.parse_config(config)
        SemanticRobotEnv.__init__(self, self.config, gpu_idx,
                                  scene_type="building",
                                  tracking_camera=tracking_camera)
        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"

        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100

        self.semantic_flagIds = []

        debug_semantic = 1
        if debug_semantic and self.gui:
            for i in range(self.semantic_pos.shape[0]):
                pos = self.semantic_pos[i]
                pos[2] += 0.2   # make flag slight above object 
                visualId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 0.7])
                flagId = p.createMultiBody(baseVisualShapeIndex=visualId, baseCollisionShapeIndex=-1, basePosition=pos)
                self.semantic_flagIds.append(flagId)

    def step(self, action):
        obs, rew, env_done, info = SemanticRobotEnv.step(self,action=action)
        self.close_semantic_ids = self.get_close_semantic_pos(dist_max=1.0, orn_max=np.pi/5)
        for i in self.close_semantic_ids:
            flagId = self.semantic_flagIds[i]
            p.changeVisualShape(flagId, -1, rgbaColor=[0, 1, 0, 1])
        return obs,rew,env_done,info

    def _rewards(self, action = None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        if self.flag_timeout > 225:
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("progress")
            print(progress)

        obstacle_penalty = 0

        #print("obs dist %.3f" %self.obstacle_dist)
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        rewards = [
            alive_score,
            progress,
            obstacle_penalty
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > 500
        if (debugmode):
            print("alive=")
            print(alive)
        #print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _reset(self):
        CameraRobotEnv._reset(self)
        for flagId in self.semantic_flagIds:
            p.changeVisualShape(flagId, -1, rgbaColor=[1, 0, 0, 1])


def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half  = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(depth[screen_half  + height_offset - screen_delta : screen_half + height_offset + screen_delta, screen_half - screen_delta : screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
       obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)
    
    debugmode = 0
    if debugmode:
        #print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty