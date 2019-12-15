import shlex
import subprocess

import cv2
import pybullet_data
from gym import error
import numpy as np
from inspect import currentframe, getframeinfo

import gibson
from gibson.core.physics.robot_locomotors import Fetch
from gibson.core.render.pcrender import PCRenderer
from gibson.data.datasets import ViewDataSet3D
from gibson.envs.env_bases import *
from gibson.envs.env_modalities import CameraRobotEnv, OneViewUI, TwoViewUI, ThreeViewUI, FourViewUI

CALC_OBSTACLE_PENALTY = 1

tracking_camera = {
    'yaw': 20,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


class FetchNavigateEnv(CameraRobotEnv):
    def _reward(self, action):
        raise NotImplementedError()

    def __init__(self, config, gpu_idx=0, depth_render_port=5556, use_filler=None):
        self.config = config
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        if isinstance(use_filler, bool):
            self._use_filler = use_filler
        else:
            self._use_filler = config["use_filler"]

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"] == "stadium" else "building",
                                tracking_camera=tracking_camera, start_port=depth_render_port, use_filler=use_filler)

        # print("Finished setting up camera_env")

        self.window_width = self.config['window_width']
        self.window_height = self.config['window_height']
        self._render_width = self.config['window_width']
        self._render_height = self.config['window_height']

        self._target_labels = self.config['target_labels']

        self.keys_to_action = {
            (ord('s'),): [-0.05, 0] + [0] * 13,  # backward
            (ord('w'),): [0.05, 0] + [0] * 13,  # forward
            (ord('d'),): [0, 0.05] + [0] * 13,  # turn right
            (ord('a'),): [0, -0.05] + [0] * 13,  # turn left
            (): [0] * 15
        }
        # print("[{} {}] Fetch init'd".format(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno))
        fetch = Fetch(self.config, env=self)
        # print("[{} {}] Introducing robot".format(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno))
        self.robot_introduce(fetch)
        # print("[{} {}] Introducing scene".format(getframeinfo(currentframe()).filename,
        #                                          getframeinfo(currentframe()).lineno))
        self.scene_introduce()
        # print("[{} {}] Scene Introduced".format(getframeinfo(currentframe()).filename,
        #                                         getframeinfo(currentframe()).lineno))
        self.total_reward = 0
        self.total_frame = 0
        self.goal_img = None
        self.initial_pos = config['initial_pos']
        self.initial_orn = config['initial_orn']
        self.step = self._step
        self.reset = self._reset

        self.nonWheelJoints = [j for j in self.robot.ordered_joints if 'wheel' not in j.joint_name]
        self.markers = []
        self.marker_ids = []

        # Initialize camera to point top down
        if self.gui:
            pos = self.robot._get_scaled_position()
            # orn = self.robot.get_orientation()
            pos = (pos[0], pos[1], pos[2] + self.tracking_camera['z_offset'])
            pos = np.array(pos)
            # dist = self.tracking_camera['distance'] / self.robot.mjcf_scaling
            # [yaw, pitch, dist] = p.getDebugVisualizerCamera()[8:11]
            p.resetDebugVisualizerCamera(3, 0, 269, pos)

    def robot_introduce(self, robot):
        self.robot = robot
        self.robot.env = self
        self.action_space = self.robot.action_space
        # Robot's eye observation, in sensor mode black pixels are returned
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        # seed for robot
        self.robot.np_random = self.np_random
        self._robot_introduced = True
        # assert (512 >= self.robot.resolution >= 64), "Robot resolution must in [64, 512]"

        self.window_width = self.config['window_width']
        self.window_height = self.config['window_height']
        self.scale_up = 1  # int(512 / self.window_width)

        self.window_dim = self.robot.resolution

        if "fast_lq_render" in self.config and self.config["fast_lq_render"]:
            self.scale_up *= 2

        self.setup_rendering_camera()

    def reset_observations(self):
        # Initialize blank render image
        self.render_rgb_filled = np.zeros((self._render_width, self._render_height, 3))
        self.render_rgb_prefilled = np.zeros((self._render_width, self._render_height, 3))
        self.render_depth = np.zeros((self._render_width, self._render_height, 1))
        self.render_normal = np.zeros((self._render_width, self._render_height, 3))
        self.render_semantics = np.zeros((self._render_width, self._render_height, 3))

    def setup_rendering_camera(self):
        if self.test_env:
            return
        self.r_camera_rgb = None  # Rendering engine
        self.r_camera_mul = None  # Multi channel rendering engine
        self.r_camera_dep = None
        # self.check_port_available()

        ui_map = {
            1: OneViewUI,
            2: TwoViewUI,
            3: ThreeViewUI,
            4: FourViewUI,
        }

        assert self.config["ui_num"] == len(
            self.config['ui_components']), "In configuration, ui_num is not equal to the number of ui components"
        if self.config["display_ui"]:
            ui_num = self.config["ui_num"]
            self.UI = ui_map[ui_num](self.window_width, self.window_height, self, self.port_ui)

        if self._require_camera_input:
            self.setup_camera_multi()
            self.setup_camera_pc()

        if self.config["mode"] == "web_ui":
            ui_num = self.config["ui_num"]
            self.webUI = ui_map[ui_num](self.window_width, self.window_height, self, self.port_ui, use_pygame=False)

    def setup_camera_pc(self):
        # Camera specific
        assert self._require_camera_input
        if self.scene_type == "building":
            self.dataset = ViewDataSet3D(
                transform=np.array,
                mist_transform=np.array,
                seqlen=2,
                off_3d=False,
                train=False,
                overwrite_fofn=True, env=self, only_load=self.config["model_id"])

        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        if not self.model_id in scene_dict.keys():
            raise error.Error("Dataset not found: models {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)

        targets, sources, source_depths, poses = [], [], [], []
        source_semantics = []

        if not self.multiprocessing or self.config["envname"] == "TestEnv":
            all_data = self.dataset.get_multi_index([v for k, v in uuids])
            for i, data in enumerate(all_data):
                target, target_depth = data[1], data[3]
                if not self._require_rgb:
                    continue
                ww = target.shape[0] // 8 + 2
                target[:ww, :, :] = target[ww, :, :]
                target[-ww:, :, :] = target[-ww, :, :]

                if self.scale_up != 1:
                    target = cv2.resize(
                        target, None,
                        fx=1.0 / self.scale_up,
                        fy=1.0 / self.scale_up,
                        interpolation=cv2.INTER_CUBIC)
                    target_depth = cv2.resize(
                        target_depth, None,
                        fx=1.0 / self.scale_up,
                        fy=1.0 / self.scale_up,
                        interpolation=cv2.INTER_CUBIC)
                pose = data[-1][0].numpy()
                targets.append(target)
                poses.append(pose)
                sources.append(target)
                source_depths.append(target_depth)
        else:
            all_data = self.dataset.get_multi_index([v for k, v in uuids])
            for i, data in enumerate(all_data):
                target, target_depth = data[1], data[3]
                if not self._require_rgb:
                    continue
                ww = target.shape[0] // 8 + 2
                target[:ww, :, :] = target[ww, :, :]
                target[-ww:, :, :] = target[-ww, :, :]

                if self.scale_up != 1:
                    target = cv2.resize(
                        target, None,
                        fx=1.0 / self.scale_up,
                        fy=1.0 / self.scale_up,
                        interpolation=cv2.INTER_CUBIC)
                    target_depth = cv2.resize(
                        target_depth, None,
                        fx=1.0 / self.scale_up,
                        fy=1.0 / self.scale_up,
                        interpolation=cv2.INTER_CUBIC)
                pose = data[-1][0].numpy()
                targets.append(target)
                poses.append(pose)
                sources.append(target)
                source_depths.append(target_depth)

        self.r_camera_rgb = PCRenderer(self.port_rgb, sources, source_depths, target, rts,
                                       scale_up=self.scale_up,
                                       semantics=source_semantics,
                                       gui=self.gui,
                                       use_filler=self._use_filler,
                                       gpu_idx=self.gpu_idx,
                                       window_width=self._render_width,
                                       window_height=self._render_height,
                                       env=self)

    def zero_joints(self):
        for j in self.ordered_joints:
            j.reset_joint_state(0, 0)

    def setup_camera_multi(self):
        assert self._require_camera_input

        def camera_multi_excepthook(exctype, value, tb):
            print("killing", self.r_camera_mul)
            self.r_camera_mul.terminate()
            if self.r_camera_dep is not None:
                self.r_camera_dep.terminate()
            if self._require_normal:
                self.r_camera_norm.terminate()
            if self._require_semantics:
                self.r_camera_semt.terminate()
            while tb:
                if exctype == KeyboardInterrupt:
                    print("Exiting Gibson...")
                    return
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print('   File "%.500s", line %d, in %.500s' % (filename, lineno, name))
                tb = tb.tb_next
            print(' %s: %s' % (exctype.__name__, value))

        sys.excepthook = camera_multi_excepthook
        enable_render_smooth = 0

        dr_path = os.path.join(os.path.dirname(os.path.abspath(gibson.__file__)), 'core', 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)

        render_main = "./depth_render --GPU {} --modelpath {} -w {} -h {} -f {} -p {}".format(self.gpu_idx,
                                                                                              self.model_path,
                                                                                              self._render_width,
                                                                                              self._render_height,
                                                                                              self.config[
                                                                                                  "fov"] / np.pi * 180,
                                                                                              self.port_depth)
        render_norm = "./depth_render --GPU {} --modelpath {} -n 1 -w {} -h {} -f {} -p {}".format(self.gpu_idx,
                                                                                                   self.model_path,
                                                                                                   self._render_width,
                                                                                                   self._render_height,
                                                                                                   self.config[
                                                                                                       "fov"] / np.pi * 180,
                                                                                                   self.port_normal)
        render_semt = "./depth_render --GPU {} --modelpath {} -t 1 -r {} -c {} -w {} -h {} -f {} -p {}".format(
            self.gpu_idx, self.model_path, self._semantic_source, self._semantic_color, self._render_width,
            self._render_height,
            self.config["fov"] / np.pi * 180, self.port_sem)

        self.r_camera_mul = subprocess.Popen(shlex.split(render_main), shell=False)
        # self.r_camera_dep = subprocess.Popen(shlex.split(render_depth), shell=False)
        if self._require_normal:
            self.r_camera_norm = subprocess.Popen(shlex.split(render_norm), shell=False)
        if self._require_semantics:
            self.r_camera_semt = subprocess.Popen(shlex.split(render_semt), shell=False)

        os.chdir(cur_path)

    def get_eye_pos_orientation(self):
        """Used in CameraEnv.setup"""
        eye_pos = self.robot.eyes.get_position()
        x, y, z, w = self.robot.eyes.get_orientation()
        eye_quat = quaternion_multiply(quaternion_multiply([w, x, y, z], [0.7071, 0.7071, 0, 0]),
                                       [0.7071, 0, -0.7071, 0]).tolist()
        return eye_pos, eye_quat

    def get_odom(self):
        return np.array(self.robot.body_xyz) - np.array(self.config["initial_pos"]), np.array(self.robot.body_rpy)

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y, z = self.robot.get_position()
        r, p, ya = self.robot.get_rpy()
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x, y, z), (10, 20), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r, p, ya), (10, 40), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _step(self, action):
        observations, reward, done, other = super()._step(action)
        # p.resetDebugVisualizerCamera(6, 0, 269, self.robot.body_xyz)
        return observations, reward, done, other

    def update_sim(self, a):
        t = time.time()
        base_obs, sensor_reward, done, sensor_meta = self._pre_update_sim(a)
        dt = time.time() - t
        # Speed bottleneck
        observations = base_obs
        self.fps = 0.9 * self.fps + 0.1 * 1 / dt

        if self.gui:
            if self.config["display_ui"]:
                self.render_to_UI()
                # print('render to ui')
                self.save_frame += 1
            elif self._require_camera_input:
                # Use non-pygame GUI
                self.r_camera_rgb.renderToScreen()

        if self.config["mode"] == 'web_ui':
            self.render_to_webUI()

        if not self._require_camera_input or self.test_env:
            return base_obs, sensor_reward, done, sensor_meta
        else:
            if self.config["show_diagnostics"] and self._require_rgb:
                self.render_rgb_filled = self.add_text(self.render_rgb_filled)

            return observations, sensor_reward, done, sensor_meta

    def _pre_update_sim(self, a):
        self.nframe += 1
        # if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            # self.robot.apply_action(a)
            # self.scene.global_step()
        p.stepSimulation()

        self.rewards = [-1]  # self._rewards(a)
        # done = self._termination()

        self.reward = 0
        self.eps_reward = 0

        if self.gui:
            pos = self.robot._get_scaled_position()
            # orn = self.robot.get_orientation()
            pos = (pos[0], pos[1], pos[2] + self.tracking_camera['z_offset'])
            pos = np.array(pos)
            # dist = self.tracking_camera['distance'] / self.robot.mjcf_scaling
            [yaw, pitch, dist] = p.getDebugVisualizerCamera()[8:11]
            p.resetDebugVisualizerCamera(dist, yaw, pitch, pos)

        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]

        # laser_scan_pos, laser_scan_quat = self.get_laser_scan_pos_orientation()
        # laser_scan_pose = [laser_scan_pos, eye_quat]

        # createPoseMarker(eye_pos, self.robot.eyes.get_orientation())
        # createPoseMarker(laser_scan_pos, laser_xyzw)

        # laser_observations = self.render_observations(laser_scan_pose)
        observations = self.render_observations(pose)
        # observations['laser'] = laser_observations['depth']

        return observations, sum(self.rewards), False, dict(eye_pos=eye_pos, eye_quat=eye_quat, episode={})

    def get_laser_scan_pos_orientation(self):
        """Used in CameraEnv.setup"""
        lpx, lpy, lpz, lrx, lry, lrz, lrw = self.robot.parts['laser_link'].get_pose()
        laser_scan_pos = np.array([lpx, lpy, lpz])
        laser_scan_quat = np.array([lrw, lrx, lry, lrz])
        # laser_scan_quat = quaternion_multiply(quaternion_multiply([lrw, lrx, lry, lrz], [0.7071, 0.7071, 0, 0]),
        #                                [0.7071, 0, -0.7071, 0]).tolist()
        return laser_scan_pos, laser_scan_quat

    def getDebugVisualizerCamera(self):
        w, h, viewMatrix, projMatrix, *_ = p.getDebugVisualizerCamera()
        w, h, rgb, d, _ = p.getCameraImage(w, h, viewMatrix, projMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return rgb

    def _rewards(self, action=None, debugmode=False):
        action_key = np.argmax(action)
        a = self.robot.action_list[action_key]
        realaction = []
        for j in self.robot.ordered_joints:
            if j.joint_name in self.robot.foot_joints.keys():
                realaction.append(self.robot.action_list[action_key][self.robot.foot_joints[j.joint_name]])
            else:
                realaction.append(0.)

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        electricity_cost = self.electricity_cost * float(
            np.abs(realaction * self.robot.joint_speeds).mean())  # let's assume we
        electricity_cost += self.stall_torque_cost * float(np.square(realaction).mean())

        steering_cost = self.robot.steering_cost(a)
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        close_to_target = 0
        # calculate min dist to a table
        min_sem_dist = 1000
        for index, semt in np.ndenumerate(self.render_semantics):
            if index == (0, 0):
                print(semt)
            if semt in self._target_labels:
                try:
                    min_sem_dist = np.min(min_sem_dist, self.render_depth[index[0]][index[1]])
                except:
                    pass
        if min_sem_dist < 2:
            close_to_target = 10

        angle_cost = self.robot.angle_cost()

        # obstacle_penalty = 0
        # if CALC_OBSTACLE_PENALTY and self._require_camera_input:
        #     obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        if debugmode:
            print("angle cost", angle_cost)

        if debugmode:
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            # print("progress")
            # print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            # print("joints_at_limit_cost")
            # print(joints_at_limit_cost)
            # print("feet_collision_cost")
            # print(feet_collision_cost)

        # rewards = [
        #     # alive,
        #     progress,
        #     # wall_collision_cost,
        #     close_to_target,
        #     steering_cost,
        #     angle_cost,
        #     obstacle_penalty
        #     # electricity_cost,
        #     # joints_at_limit_cost,
        #     # feet_collision_cost
        # ]
        return [-1 if len(wall_contact) > 0 else 1]

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))

        done = not alive or self.nframe > 2000
        # if done:
        #    print("Episode reset")
        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH,
                                                     fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                     meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1,
                                                 basePosition=[target_pos[0], target_pos[1], 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()

        [p.removeBody(i) for i in self.marker_ids]
        self.markers = []
        self.marker_ids = []
        [j.reset_joint_state(0, 0) for j in self.nonWheelJoints]
        return obs

    def reset_joints(self):
        [j.reset_joint_state(0, 0) for j in self.nonWheelJoints]

    def hide_pose_markers(self):
        [p.removeBody(i) for i in self.marker_ids]
        self.marker_ids = []

    def add_pose_marker(self, position=None, color="red"):
        if position is None:
            position = self.robot.body_xyz

        self.markers.append((color, position))

    def load_pose_markers(self, size=0.25):
        """Create a pose marker that identifies a position and orientation in space with 3 colored lines.
        """
        [p.removeBody(i) for i in self.marker_ids]
        self.marker_ids = []

        for color, pos in self.markers:
            vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
            self.marker_ids.append(p.createMultiBody(basePosition=pos, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id))

    def visualize_trajectory(self, positions, size=0.25, color="red"):
        [p.removeBody(i) for i in self.marker_ids]
        self.marker_ids = []
        for position in positions:
            vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
            self.marker_ids.append(p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id))


def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(
        depth[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
        screen_half - screen_delta: screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
        obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)

    debugmode = 0
    if debugmode:
        # print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty
