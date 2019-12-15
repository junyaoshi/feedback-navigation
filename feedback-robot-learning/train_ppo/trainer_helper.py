import os
import pickle
import numpy as np
import pybullet as p

USE_WALL_CONTACT = True

def run_dijkstra(env, target_position, goal_tolerance=0.5, use_other_room=False):
    """
    Run dijkstra in 3D space
    target_position = (goal_x, goal_y, goal_z)
    Assumptions:
        x_range1: -16, -13.5
        y range1: 16.3, 26.7
        x range2: -13.5, -5
        y range2: 18.9, 27.2
        yaw range: 0, 2pi
        x resolution: 0.05
        y resolution: 0.05
        yaw resolution: pi/6

        action length:
            forward/backward: 0.5
            left/right: pi/6
    """
    env._reset_fov()

    # check space w/o collision

    # for experiment room
    if use_other_room:
        x_range1 = [-23.4, -14]
        y_range1 = [47.5, 50.7]
        x_range2 = x_range1
        y_range2 = y_range1
        safe_range = [[[-22.9, -14.5], [48, 50.2]]]
    else:
        x_range1 = [-16.3, -13.5]
        y_range1 = [16.0, 26.7]
        x_range2 = [-13.5, -4.6]
        y_range2 = [18.9, 27.2]
        safe_range = [[[-15.9, -14.64], [16.16, 26.6]], [[-11.2, -8.43], [20.56, 23.58]], [[-11.2, -7.74], [20.56, 22.74]]]

    # for other room
    x_resolution = y_resolution = 0.02

    goal_pos = target_position[:2]

    x_overall_range = [np.min(x_range1 + x_range2), np.max(x_range1 + x_range2)]
    y_overall_range = [np.min(y_range1 + y_range2), np.max(y_range1 + y_range2)]

    yaw_range = [0, 2 * np.pi]
    yaw_resolution = np.pi / 6

    shape = (int(round((x_overall_range[1] - x_overall_range[0]) / x_resolution)) + 1,
             int(round((y_overall_range[1] - y_overall_range[0]) / y_resolution)) + 1,
             int((yaw_range[1] - yaw_range[0]) / yaw_resolution))

    def idx_2_cor(x_idx, y_idx, yaw_idx):
        x = x_overall_range[0] + x_idx * x_resolution
        y = y_overall_range[0] + y_idx * y_resolution
        yaw = yaw_range[0] + yaw_idx * yaw_resolution
        return x, y, yaw

    def cor_2_idx(x, y, yaw):
        x_idx = int(round((x - x_overall_range[0]) / x_resolution))
        y_idx = int(round((y - y_overall_range[0]) / y_resolution))
        yaw_idx = int(round((yaw - yaw_range[0]) / yaw_resolution))
        if yaw_idx == 12:
            yaw_idx = 0
        return x_idx, y_idx, yaw_idx

    def clear_region(collision, x_range, y_range):
        res = np.copy(collision)
        x_min_idx = int(round((x_range[0] - x_overall_range[0]) / x_resolution))
        x_max_idx = int(round((x_range[1] - x_overall_range[0]) / x_resolution)) + 1
        y_min_idx = int(round((y_range[0] - y_overall_range[0]) / y_resolution))
        y_max_idx = int(round((y_range[1] - y_overall_range[0]) / y_resolution)) + 1
        res[x_min_idx:x_max_idx, y_min_idx:y_max_idx, :] = \
            np.full((x_max_idx - x_min_idx, y_max_idx - y_min_idx, shape[2]), False)
        return res

    def euclidean_2d_dist(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_neighbors(x_idx, y_idx, yaw_idx):
        neighbors = []
        x, y, yaw = idx_2_cor(x_idx, y_idx, yaw_idx)
        for i in range(0, env.action_space.n):
            neighbor_idx, met_collision = get_neighbor_from_action(x, y, yaw, i, reverse=env._exclude_backward)
            if not env._exclude_backward:
                if i == 0:
                    cost = 1000000
                elif i == 1:
                    cost = 1
                else:
                    cost = 2
            else:
                if i == 0:
                    cost = 1
                else:
                    cost = 2
            if not met_collision:
                neighbors.append((neighbor_idx, cost))
        return neighbors

    def get_neighbor_from_action(x, y, yaw, action,
                                 step_len=0.3, step_angle=np.pi/6, collision_check_resolution=0.01,
                                 reverse=False):
        # x, y, yaw = idx_2_cor(x_idx, y_idx, yaw_idx)
        met_collision = False
        if env._exclude_backward and action != 0:
            action += 1
        if action == 0 or action == 1:
            new_yaw = yaw
            for step in np.arange(0, step_len, collision_check_resolution):
                step = min(step + collision_check_resolution, step_len)
                if (action == 0 and not reverse):  # Foward
                    new_x = x + step * np.cos(yaw)
                    new_y = y + step * np.sin(yaw)
                else:  # Backward
                    new_x = x - step * np.cos(yaw)
                    new_y = y - step * np.sin(yaw)
                new_x_idx, new_y_idx, new_yaw_idx = cor_2_idx(new_x, new_y, new_yaw)
                if new_x_idx < 0 or new_x_idx >= shape[0] or new_y_idx < 0 or new_y_idx >= shape[1] \
                    or collision[new_x_idx, new_y_idx, new_yaw_idx]:
                    met_collision = True
                    break
        elif action == 2 or action == 3:
            new_x = x
            new_y = y
            if action == 3:
                new_yaw = yaw + step_angle
                if new_yaw >= 2 * np.pi:
                    new_yaw -= 2 * np.pi
            else:
                new_yaw = yaw - step_angle
                if new_yaw < 0:
                    new_yaw += 2 * np.pi
            new_x_idx, new_y_idx, new_yaw_idx = cor_2_idx(new_x, new_y, new_yaw)
            if new_x_idx < 0 or new_x_idx >= shape[0] or new_y_idx < 0 or new_y_idx >= shape[1] \
                or collision[new_x_idx, new_y_idx, new_yaw_idx]:
                met_collision = True
            else:
                met_collision = collision[new_x_idx, new_y_idx, new_yaw_idx]
        else:
            raise ValueError("Unknown action: {}".format(action))
        if not env._exclude_backward and action == env.BACKWARD:
            met_collision = True
        return (new_x_idx, new_y_idx, new_yaw_idx), met_collision

    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dijkstra_dir = "dijkstra_other_room" if use_other_room else "dijkstra_via_speed"
    dijkstra_fname = os.path.join(currentdir, dijkstra_dir, "husky_navigate_2D_dijkstra.p")

    print(dijkstra_fname)
    if os.path.exists(dijkstra_fname):
        with open(dijkstra_fname, "rb") as f:
            dijkstra_data = pickle.load(f)
    else:
        dijkstra_data = {}

    if "collision" in dijkstra_data:
        collision = dijkstra_data["collision"]
        print("collision matrix successfully loaded")
    else:
        print("initiating collision detection sequence")
        collision = np.full(shape, True)
        collision = clear_region(collision, x_range1, y_range1)
        collision = clear_region(collision, x_range2, y_range2)

        safe = np.full(shape, True)
        for (x_range, y_range) in safe_range:
            safe = clear_region(safe, x_range, y_range)
        safe = np.invert(safe)

        from transforms3d.euler import euler2quat
        from gibson.core.physics.robot_bases import quatToXYZW

        dim1, dim2, dim3 = collision.shape
        collision_total_num = dim1 * dim2 * dim3

        collision_progress = 0
        collision_num = 0
        for x_idx in range(shape[0]):
            print("collision checking progress: {}".format(collision_progress / collision_total_num))
            if collision_progress != 0:
                print("percentage of collision nodes: {}".format(round(collision_num / collision_progress, 5)))
            for y_idx in range(shape[1]):
                for yaw_idx in range(shape[2]):
                    collision_progress += 1
                    if collision[x_idx, y_idx, yaw_idx]:
                        continue
                    if safe[x_idx, y_idx, yaw_idx]:
                        continue
                    x, y, yaw = idx_2_cor(x_idx, y_idx, yaw_idx)
                    env.robot.robot_body.reset_position([x, y, 0.15])
                    env.robot.robot_body.reset_orientation(quatToXYZW(euler2quat(*[0, 0, yaw]), 'wxyz'))
                    # env._reset_fov()
                    if not USE_WALL_CONTACT:
                        env.step(env.IDLE)
                        vx, vy, _ = env.robot.robot_body.speed()
                        collision[x_idx, y_idx, yaw_idx] = np.sqrt(vx**2 + vy**2) > 0.08
                    else:
                        env.step(env.IDLE)
                        # print('x: {}, y: {}, yaw: {}, in collision: {}'.format(x, y, yaw, env.in_collision()))
                        collision[x_idx, y_idx, yaw_idx] = env.in_collision()
                        if env.in_collision():
                            collision_num += 1

        dijkstra_data["collision"] = collision
        with open(dijkstra_fname, "wb") as f:
            pickle.dump(dijkstra_data, f)

    if False:
        pass
    if "dist" in dijkstra_data:
        dist = dijkstra_data["dist"]
        print("dist matrix successfully loaded")
    else:
        print("initiating dijkstra")
        # dijkstra
        from heapq import heappush, heappop

        dist = np.full(shape, np.inf)
        visited = collision.copy()
        priority_q = []
        for x_idx in range(shape[0]):
            for y_idx in range(shape[1]):
                for yaw_idx in range(shape[2]):
                    if collision[x_idx, y_idx, yaw_idx]:
                        continue
                    x, y, yaw = idx_2_cor(x_idx, y_idx, yaw_idx)
                    if euclidean_2d_dist(goal_pos, (x, y)) < goal_tolerance:
                        dist[x_idx, y_idx, yaw_idx] = 0.0
                    heappush(priority_q, (dist[x_idx, y_idx, yaw_idx], (x_idx, y_idx, yaw_idx)))

        while priority_q:
            node_dist, (node_x_idx, node_y_idx, node_yaw_idx) = heappop(priority_q)
            if visited[node_x_idx, node_y_idx, node_yaw_idx]:
                continue
            visited[node_x_idx, node_y_idx, node_yaw_idx] = True
            for neighbor_idx, cost in get_neighbors(node_x_idx, node_y_idx, node_yaw_idx):
                neighbor_x_idx, neighbor_y_idx, neighbor_yaw_idx = neighbor_idx
                if visited[neighbor_x_idx, neighbor_y_idx, neighbor_yaw_idx]:
                    continue
                tmp = node_dist + cost
                if tmp < dist[neighbor_x_idx, neighbor_y_idx, neighbor_yaw_idx]:
                    dist[neighbor_x_idx, neighbor_y_idx, neighbor_yaw_idx] = tmp
                    # visited[node_x_idx, node_y_idx, node_yaw_idx] = False
                    heappush(priority_q, (tmp, (neighbor_x_idx, neighbor_y_idx, neighbor_yaw_idx)))

        dijkstra_data["dist"] = dist
        print("dijkstra ran successfully")
        with open(dijkstra_fname, "wb") as f:
            pickle.dump(dijkstra_data, f)

    def judge_action(state, action):
        x, y, yaw = state
        dists = []
        idxs = cor_2_idx(x, y, yaw)
        for i in range(env.action_space.n):
            (new_x_idx, new_y_idx, new_yaw_idx), met_collision = get_neighbor_from_action(x, y, yaw, i)
            if met_collision:
                dists.append(np.inf)
            else:
                neighbor_dist = dist[new_x_idx, new_y_idx, new_yaw_idx]
                self_dist = dist[idxs]
                if neighbor_dist >= self_dist:
                    dists.append(np.inf)
                else:
                    dists.append(neighbor_dist)

        # debugging
        # print("loc", state, "action", action)
        # print('dist of self: {}'.format(self_dist))
        # print('dists of neighbors: {}'.format(dists))

        if min(dists) == np.inf:
            return False
        if dists[env.FORWARD] == min(dists):
            return action == env.FORWARD
        return dists[action] == min(dists)

    def get_good_actions(state):
        good_actions = []
        x, y, yaw = state
        dists = []
        idxs = cor_2_idx(x, y, yaw)
        for i in range(env.action_space.n):
            (new_x_idx, new_y_idx, new_yaw_idx), met_collision = get_neighbor_from_action(x, y, yaw, i)
            if met_collision:
                # print('action {} has dist {}'.format(i, 'met collision'))
                dists.append(np.inf)
            else:
                neighbor_dist = dist[new_x_idx, new_y_idx, new_yaw_idx]
                self_dist = dist[idxs]
                if neighbor_dist >= self_dist:
                    # print('action {} has dist {}'.format(i, neighbor_dist))
                    dists.append(np.inf)
                else:
                    # print('action {} has dist {}'.format(i, neighbor_dist))
                    dists.append(neighbor_dist)

        # debugging
        # print("loc", state, "action", action)
        # print('dist of self: {}'.format(self_dist))

        min_dist = min(dists)
        if min_dist == np.inf:
            # print('all actions are inf')
            pass
        else:
            for action in range(env.action_space.n):
                if dists[action] == min_dist:
                    if action == env.FORWARD:
                        good_actions.append(action)
                        break
                    else:
                        good_actions.append(action)
        return good_actions

    def generate_shortest_paths(state):
        x, y, yaw = state
        dists = []
        paths = []
        init_nodes = []
        for i in range(env.action_space.n):
            (new_x_idx, new_y_idx, new_yaw_idx), met_collision = get_neighbor_from_action(x, y, yaw, i)
            init_nodes.append([new_x_idx, new_y_idx, new_yaw_idx])
            if met_collision:
                dists.append(np.inf)
            else:
                dists.append(dist[new_x_idx, new_y_idx, new_yaw_idx])

        if min(dists) == np.inf:
            return []

        for i in range(env.action_space.n):
            if dists[i] == min(dists):
                x_node_idx, y_node_idx, yaw_node_idx = init_nodes[i]
                x_node, y_node, yaw_node = idx_2_cor(x_node_idx, y_node_idx, yaw_node_idx)
                path = [state, [x_node, y_node, yaw_node]]
                path_valid = True
                while (dist[x_node_idx, y_node_idx, yaw_node_idx] > 0):
                    # print(x_node_idx, y_node_idx, yaw_node_idx, dist[x_node_idx, y_node_idx, yaw_node_idx])
                    nodes = []
                    node_dists = []
                    for j in range(env.action_space.n):
                        (new_x_idx, new_y_idx, new_yaw_idx), met_collision = get_neighbor_from_action(x_node, y_node,
                                                                                                      yaw_node, j)
                        nodes.append([new_x_idx, new_y_idx, new_yaw_idx])
                        if met_collision:
                            node_dists.append(np.inf)
                        else:
                            node_dists.append(dist[new_x_idx, new_y_idx, new_yaw_idx])
                    if min(node_dists) == np.inf or min(node_dists) >= dist[x_node_idx, y_node_idx, yaw_node_idx]:
                        path_valid = False
                        break
                    # print("neighbors", node_dists, dist[x_node_idx, y_node_idx, yaw_node_idx])
                    min_idxs = []
                    for j in range(env.action_space.n):
                        if node_dists[j] == min(node_dists):
                            min_idxs.append(j)
                    idx = np.random.choice(min_idxs)
                    x_node_idx, y_node_idx, yaw_node_idx = nodes[idx]
                    x_node, y_node, yaw_node = idx_2_cor(x_node_idx, y_node_idx, yaw_node_idx)
                    path.append([x_node, y_node, yaw_node])
                if path_valid:
                    paths.append(path)
        return paths

    def check_collision(state):
        x_max, y_max, _ = collision.shape
        x, y, yaw = state
        x_idx, y_idx, yaw_idx = cor_2_idx(x, y, yaw)
        if x_idx <= 0 or x_idx >= x_max or y_idx <= 0 or y_idx >= y_max:
            return False
        return collision[x_idx, y_idx, yaw_idx]

    return judge_action, get_good_actions, generate_shortest_paths, get_neighbor_from_action, idx_2_cor, check_collision


def judge_action_1D(target_position):
    _, y_target, _ = target_position
    def judge_action(state, action):
        y_now = state[0]
        return (y_now >= y_target and action == 0) or (y_now < y_target and action == 1)
    return judge_action


def get_simulated_feedback(obs, actions, action_idxs, judge_action, good_acc=0.7, bad_acc=0.7):
    feedbacks = []
    correct_feedbacks = []
    for s, a in zip(obs, actions):
        good_move = judge_action(s, a)
        if good_move:
            feedback = np.random.uniform() < good_acc
        else:
            feedback = np.random.uniform() > bad_acc
        feedbacks.append(int(feedback))
        correct_feedbacks.append(int(good_move))
    return action_idxs, np.asarray(feedbacks), np.asarray(correct_feedbacks)


def get_feedback_from_LSL(feedback_stream):
    feedback_samples, feedback_timestamps = feedback_stream.pull_chunk()
    feedback = np.array([ele[0] for ele in feedback_samples])
    action_idx = np.array([ele[2] for ele in feedback_samples])
    return action_idx, feedback, np.copy(feedback)