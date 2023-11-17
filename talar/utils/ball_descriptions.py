
import os
import numpy as np


BEHIND = 0  # +y
LEFT = 1  # -x
FRONT = 2  # -y
RIGHT = 3  # +x
dir_list = [
    BEHIND,
    LEFT,
    FRONT,
    RIGHT,
]
dir_cnt = len(dir_list)
step_size = 360 / dir_cnt
rg_size = step_size / 2
dir2angle_rg = {
    0: np.array([90 - rg_size, 90 + rg_size]),
}
for dir in range(1, dir_cnt):
    dir2angle_rg[dir] = (dir2angle_rg[dir - 1] + step_size) % 360


import sys
sys.path.append(os.path.join(os.getcwd(), "envs"))
sys.path.append(os.path.join(os.path.join(os.getcwd(), "envs"), "clevr_robot_env"))


color_order = ["red", "blue", "green", "purple", "cyan"]

template_list = [
    "Push the {} ball {} the {} ball",              
    "Can you push the {} ball {} the {} ball",
    "Can you help me push the {} ball {} the {} ball",
    "Is the {} ball {} the {} ball",
    "Is there any {} ball {} the {} ball",
    "The {} ball moves {} the {} ball",
    "The {} ball is being pushed {} the {} ball",
    "The {} ball is pushed {} the {} ball",
    "The {} ball was moved {} the {} ball",

    "Move the {} ball {} the {} ball",             
    "Keep the {} ball {} the {} ball",
    "Can you move the {} ball {} the {} ball",
    "Can you keep the {} ball {} the {} ball",
    "Can you help me move the {} ball {} the {} ball",
    "Can you help me keep the {} ball {} the {} ball",
    "The {} ball was pushed {} the {} ball",
    "The {} ball is being moved {} the {} ball",
    "The {} ball is moved {} the {} ball",
]

orientation_list = ["behind", "to the left of", "in front of", "to the right of"]


def compute_angle(src_xy: np.ndarray, dst_xy: np.ndarray = None):
    epsilon = 1e-3
    delta_xy = src_xy - dst_xy
    delta_xy = np.where(np.abs(delta_xy) >= epsilon, delta_xy, 0)
    cos_value = delta_xy[0] / np.linalg.norm(delta_xy)
    negative_flag = delta_xy[1] < 0
    angle = np.arccos(cos_value)
    if negative_flag:
        angle = 2 * np.pi - angle
    _angle = angle * 180 / np.pi
    return _angle


def calculate_description(next_obs: np.ndarray, goal: np.ndarray, data_type="new") -> int:
    goal_src = 1
    goal_dst = 2
    num_object = 5
    shape_obs = next_obs.reshape(num_object, -1)
    
    if data_type == "new":
        src_idx = int(np.where(goal[:num_object] == goal_src)[0])
        dst_idx = int(np.where(goal[:num_object] == goal_dst)[0])
    else:
        src_idx = int(np.where(goal[:num_object] == 1)[0][0])
        dst_idx = int(np.where(goal[:num_object] == 1)[0][-1])
    src_xy = shape_obs[src_idx]
    dst_xy = shape_obs[dst_idx]
    
    curr_angle = compute_angle(src_xy, dst_xy)
    orientation = None
    for dir in range(dir_cnt):
        angle_rg = dir2angle_rg[dir]
        if angle_rg[1] < angle_rg[0]:
            angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
        else:
            angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
        if angle_flag:
            orientation = dir
    assert orientation is not None
    
    return src_idx, dst_idx, orientation


def calculate_direction(src_xy, dst_xy):
    curr_angle = compute_angle(src_xy, dst_xy)
    orientation = None
    for dir in range(dir_cnt):
        angle_rg = dir2angle_rg[dir]
        if angle_rg[1] < angle_rg[0]:
            angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
        else:
            angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
        if angle_flag:
            orientation = dir
    return orientation


def get_balls_description(goal, obs, next_obs, t):
    pushed, target, orientation = calculate_description(next_obs, goal)
    description = template_list[t].format(color_order[pushed], orientation_list[orientation], color_order[target])
    return description

