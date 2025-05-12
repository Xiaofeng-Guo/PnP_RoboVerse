from __future__ import annotations

import math

from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass


import torch

from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass
from metasim.utils.state import TensorState

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass



def negative_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    for i, name in enumerate(states.robots[robot_name].body_names):
        print(f"{i}: {name}")
        print("pos", states.robots[robot_name].body_state[0, i, :3])
    print("ee_pos", ee_pos)
    distances = torch.norm(ee_pos, dim=1)
    return -distances  # Negative distance as reward

def x_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    return ee_pos[:, 0]


def obj_height(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    obj_pos = states.objects["block_red"].root_state[:, :3]
    reward = torch.zeros(obj_pos.shape[0], device=obj_pos.device)



    # check if the object is in the hand
    obj_pos = states.objects["block_red"].root_state[:, :3]
    # print("obj_pos", obj_pos[0])
    # get the finger position
    left_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_leftfinger"), :3]
    right_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_rightfinger"), :3]
    y_left = left_finger_pos[:, 1] - obj_pos[:, 1]
    y_right = right_finger_pos[:, 1] - obj_pos[:, 1]
    y_in_hand = torch.multiply(y_left, y_right)

    z_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 2]
    z_dis = torch.abs(z_hand - obj_pos[:, 2])

    x_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 0]
    x_dis = torch.abs(x_hand - obj_pos[:, 0])

    # use logic to check if the object is in the hand
    in_hand = torch.logical_and(x_dis < 0.02, torch.logical_and(y_in_hand < 0, torch.logical_and(z_dis>0.108, z_dis < 0.125)))





    for i in range(obj_pos.shape[0]):
        # if the object is too low, set a large negative reward
        if obj_pos[i, 2] < 0.198:
            reward[i] += -1
        # if the object is too high, set a large negative reward
        elif obj_pos[i, 2] < 0.2:
            reward[i] += 0.0
        elif obj_pos[i, 2] < 0.22:
            if in_hand[i]:
                # if the object is in the hand, set the reward to be 1
                reward[i] += (obj_pos[i, 2]-0.2)*100
            else:
                reward[i] += 0.0
        else:
            reward[i] += (obj_pos[i, 2]-0.2)*100
        if obj_pos[i, 2] > 0.204:
            if in_hand[i]:
                reward[i] += 1.0
            else:
                reward[i] += 0.0
        if obj_pos[i, 2] > 0.21:
            reward[i] += 0.2
        if obj_pos[i, 2] > 0.22:
            reward[i] += 0.5
    print("obj_height", obj_pos[0, 2])
    reward = torch.clamp(reward, -1, 20)
    print("reward", reward[0])
    return reward

def naive_obj_hand_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    obj_pos = states.objects["block_red"].root_state[:, :3]
    obj_pos[:, 2] += 0.125
    # Calculate the distance between the end effector and the object
    distances = torch.norm(ee_pos[:,:3] - obj_pos[:,:3], dim=1)
    # Use negative distance as reward
    reward = -distances
    # uniformly scale the reward to be between -1 and 1
    reward = reward * 2 + 1.0

    return reward


def obj_hand_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    obj_pos = states.objects["block_red"].root_state[:, :3]

    reward = torch.zeros(ee_pos.shape[0], device=ee_pos.device)

    xy_distance = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=1)
    print("xy_distance", xy_distance[0])

    z_distance = ee_pos[:, 2] - obj_pos[:, 2]
    print("z_distance", z_distance[0])

    for i in range(xy_distance.shape[0]):
        # if the object and hand is far away in x-y plane, encourage the hand to move to the correct x-y plane while encourage the z in a distance
        if xy_distance[i] > 0.05:
            reward[i] += -xy_distance[i]
            z_target_distance = z_distance[i] - 0.125
            if z_target_distance < 0.02:
                z_target_distance = 0
            reward[i] += -z_target_distance
        # if the object and hand is close in x-y plane, encourage the hand to move to the correct z distance
        # elif xy_distance[i] > 0.001:
        else:
            reward[i] += 0.5
            reward[i] += -xy_distance[i]
            z_target_distance = z_distance[i]- 0.125
            if z_target_distance < 0.02:
                z_target_distance = 0
            reward[i] += -z_target_distance
        if z_distance[i] <0.11:
            reward[i] -=0.5

        # else:
        #     reward[i] += 20
        #     reward[i] += -xy_distance[i] * 20
        #     z_target_distance = torch.abs(z_distance[i])
        #     if z_target_distance < 0.10:
        #         z_target_distance = 0
        #     reward[i] += -(z_target_distance**2)*5
    # print("reward", reward)
    # print("joint pos", states.robots[robot_name].joint_pos[:, :])
    # reward = reward
    reward = torch.clamp(reward, -1, 1)
    print("reward", reward[0])
    return reward

def fingerip_obj_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the position of the object
    obj_pos = states.objects["block_red"].root_state[:, :3]
    # print the body names

    # for i, name in enumerate(states.robots[robot_name].body_names):
    #     print(f"{i}: {name}")


    left_finger_pos= states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_leftfinger"), :3]
    right_finger_pos= states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_rightfinger"), :3]
    ee_pos = (left_finger_pos + right_finger_pos)/2
    hand_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]

    # Calculate the x_y distance between the object and the end effector
    xy_distance = torch.norm(ee_pos[:, :2] - hand_pos[:, :2], dim=1)
    # fingerbase_obj_distance = torch.norm(ee_pos - obj_pos, dim=1)
    # print("ee_pos", ee_pos)
    # print("hand_pos", hand_pos)
    # print("xy_distance", xy_distance)

    reward = torch.zeros(ee_pos.shape[0], device=ee_pos.device)
    reward += - xy_distance * 1000
    # reward += - fingerbase_obj_distance

    return reward

def fine_obj_hand_distance(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    obj_pos = states.objects["block_red"].root_state[:, :3]
    distance = torch.norm(ee_pos - obj_pos, dim=1)
    distance[distance < 0.001] = 0
    reward = -(distance**2)*10
    # clip the reward to be between -1 and 1
    reward = torch.clamp(reward, -1, 1)
    # print("reward", reward)
    return reward



def hand_orientation(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the orientation of the end effector
    ee_orientation = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 3:7]
    # print("ee_orientation", ee_orientation[0])
    left_finger_orientation = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_leftfinger"), 3:7]
    right_finger_orientation = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_rightfinger"), 3:7]
    # Calculate the reward based on the orientation
    # get the different between the end effector orientation and the target orientation
    # move to device
    target_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(ee_orientation.device)

    eps=1e-8
    d = torch.abs(torch.sum(ee_orientation * target_orientation, dim=1))
    d = torch.clamp(d, -1.0 + eps, 1.0 - eps)          # numerical safety
    theta = 2.0 * torch.arccos(d)

    # set the theta to be between 0 and 1
    theta = theta/3.1416
    # print("theta", theta[0])
    scale = 10.0
    f_theta = (1-theta) ** 2
    # print("theta", f_theta[0])
    # normalize the theta to be between 0 and 1
    reward = (torch.exp(scale * (f_theta))-1)/(torch.exp(torch.tensor(scale)) - 1.0)
    reward[theta<0.02] += 0.5
    # print("reward", reward[0])

    # print("dot_product", dot_product)
    # reward = 0
    # reward += dot_product**2
    # # print("dot_product", dot_product)
    # if the dot product is less than 0.98, set a large negative reward
    # dot_product[dot_product < 0.7] = -2
    # dot_product[(dot_product > 0) & (dot_product < 0.8)] = -1.5
    # dot_product[(dot_product > 0.0) & (dot_product < 0.9)] = -1.0
    # dot_product[(dot_product > 0.0) & (dot_product < 0.995)] = -0.8
    # dot_product[(dot_product>0) & (dot_product < 0.995)] = -0.5
    # # if the dot product is greater than 0.98, make it squared
    # # print("right_finger_orientation", right_finger_orientation)
    # dot_product[dot_product > 0.995] = ((dot_product[dot_product > 0.995]-0.995)/(0.005))
    # reward += dot_product
    # print("reward", reward)

    return reward

def hand_target(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # get the hand position
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    # set the target position
    target_pos = torch.tensor([0.0, 0.0, 1.0]).to(ee_pos.device)
    # calculate the distance between the hand and the target
    distances = torch.norm(ee_pos - target_pos, dim=1)
    reward = -distances
    # print("reward", reward)
    return reward

def obj_position(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the position of the object
    obj_pos = states.objects["block_red"].root_state[:, :3]
    # set the initial position of the object
    initial_pos = torch.tensor([0.25, 0.0, 0.0]).to(obj_pos.device)
    # Calculate the distance between the object and the initial position of the x,y axis only
    distances = torch.norm(obj_pos[:, :2] - initial_pos[:2], dim=1)
    reward = -(distances)*50
    # clip the reward to be between -1 and 1
    reward = torch.clamp(reward, -1, 1)

    return reward

def finger_width (states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # make the finger width to be close to the object width
    # get the finger width
    left_finger_width = states.robots[robot_name].joint_pos[:, 7]
    right_finger_width = states.robots[robot_name].joint_pos[:, 8]
    # finger_width = (left_finger_width + right_finger_width)
    # get the object width
    obj_width = 0.05
    # calculate the distance between the finger width and the object width
    left_distances = left_finger_width - obj_width/2
    right_distances = right_finger_width - obj_width/2
    print("left_distances", left_distances[0])
    print("right_distances", right_distances[0])

    left_distances[left_distances < -0.02] = -0.02
    right_distances[right_distances < -0.02] = -0.02
    # use negative distance as reward
    distances = left_distances + right_distances
    # reward = -distances * 10


    # check if the object is in the hand
    obj_pos = states.objects["block_red"].root_state[:, :3]
    # print("obj_pos", obj_pos[0])
    # get the finger position
    left_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_leftfinger"), :3]
    right_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_rightfinger"), :3]
    y_left = left_finger_pos[:, 1] - obj_pos[:, 1]
    y_right = right_finger_pos[:, 1] - obj_pos[:, 1]
    y_in_hand = torch.multiply(y_left, y_right)

    z_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 2]
    z_dis = torch.abs(z_hand - obj_pos[:, 2])

    x_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 0]
    x_dis = torch.abs(x_hand - obj_pos[:, 0])

    # use logic to check if the object is in the hand
    in_hand = torch.logical_and(x_dis < 0.02, torch.logical_and(y_in_hand < 0, torch.logical_and(z_dis>0.11, z_dis < 0.125)))
    reward = torch.zeros(in_hand.shape[0], device=in_hand.device)
    for i in range(in_hand.shape[0]):
        if not in_hand[i]:
            if obj_pos[i, 2] < 0.21:
                reward[i] = (left_finger_width[i] + right_finger_width[i]) * 25.0 - 1.5
            else:
                reward[i] += 5
                reward[i] += -distances[i] * 100
        else:
            # if the object is in the hand, set the reward to be 0
            reward[i] += 5
            if distances[i] < -0.01:
                reward[i]+=2
            reward[i] += -distances[i] * 100
    print("x_in_hand", x_dis[0])
    print("y_in_hand", y_in_hand[0])
    print("z_in_hand", z_dis[0])
    print("in_handornot", in_hand[0])
    print("finger_width reward", reward[0])
    reward = reward/10.0
    # clip the reward to be between -1 and 1
    reward = torch.clamp(reward, -1, 1)
    return reward

def obj_in_hand (states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # check whether the object is in the hand
    # get the object position
    obj_pos = states.objects["block_red"].root_state[:, :3]
    # get the finger position
    left_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_leftfinger"), :3]
    right_finger_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_rightfinger"), :3]

    # check if the obj pos y aixs is within the finger pos y axis
    y_left = left_finger_pos[:, 1] - obj_pos[:, 1]
    y_right = right_finger_pos[:, 1] - obj_pos[:, 1]
    y_in_hand = torch.multiply(y_left, y_right)

    z_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 2]
    z_dis = torch.abs(z_hand - obj_pos[:, 2])
    x_hand = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 0]
    x_dis = torch.abs(x_hand - obj_pos[:, 0])

    # use logic to check if the object is in the hand
    in_hand = torch.logical_and(x_dis < 0.02, torch.logical_and(y_in_hand < 0, torch.logical_and(z_dis>0.11, z_dis < 0.125)))

    # if in hand, set the reward to be 1
    # if not in hand, set the reward to be -1
    reward = torch.zeros(in_hand.shape[0], device=in_hand.device)
    for i in range(in_hand.shape[0]):
        if not in_hand[i]:
            reward[i] = -1
        if in_hand[i]:
            # if the object is in the hand, set the reward to be 1
            reward[i] = 1
    print("in_hand reward", in_hand[0])
    # clip the reward to be between 0 and 1
    reward = torch.clamp(reward, -1,1)
    return reward

def tiny_ee_vel (states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the velocity of the end effector
    ee_vel = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), 7:10]
    # Calculate the reward based on the velocity
    vel = torch.norm(ee_vel, dim=1)
    reward = -vel**2
    # print("ee_vel", ee_vel)
    # print("vel", vel)
    return reward


def reward_joint_angle_0(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 0]
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward

def reward_joint_angle_1(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 1]
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward
def reward_joint_angle_2(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 2]
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward
def reward_joint_angle_3(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 3]
    # print("joint_angles", joint_angles[0])
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward
def reward_joint_angle_4(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 4]
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward
def reward_joint_angle_5(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 5]
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 10.0
    reward = -torch.exp(scale * (joint_angles - target_joint_angle)**2)+2

    return reward

def reward_joint_angle_6(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    # Get the joint angles of the robot
    joint_angles = states.robots[robot_name].joint_pos[:, 6]
    # print("joint_angles", joint_angles)
    # Calculate the reward based on the joint angles
    target_joint_angle = 0.0
    scale = 20.0
    reward = -torch.abs(joint_angles - target_joint_angle)*100.0 + 1
    return reward





@configclass
class BinPickingCfg(BaseTaskCfg):
    episode_length = 250
    # objects = [
    #     ArticulationObjCfg(
    #         name="box_base",
    #         usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
    #         urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
    #         mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
    #     ),
    # ]

    objects = [
        PrimitiveCubeCfg(
            name="block_red",
            mass=0.1,  # origin is 1, smaller mass is easier to grasp
            size=(0.1, 0.05, 0.1),
            color=(1.0, 0.0, 0.0),
            physics=PhysicStateType.RIGIDBODY,
            scale=0.8,
        ),
        PrimitiveCubeCfg(
            name="block_black",
            mass=10000.0,  # origin is 1, smaller mass is easier to grasp
            size=(1.0, 1.0, 0.2),
            color=(0.0, 0.0, 0.0),
            physics=PhysicStateType.RIGIDBODY,
            scale=0.8,
        ),
        # PrimitiveCubeCfg(
        #     name="block_blue",
        #     mass=0.1,  # origin is 1, smaller mass is easier to grasp
        #     size=(0.1, 0.05, 0.05),
        #     color=(0.0, 0.0, 1.0),
        #     physics=PhysicStateType.RIGIDBODY,
        #     scale=0.8,
        # ),
        # PrimitiveCubeCfg(
        #     name="block_pink",
        #     mass=0.1,  # origin is 1, smaller mass is easier to grasp
        #     size=(0.05, 0.05, 0.05),
        #     color=(1.0, 0.0, 1.0),
        #     physics=PhysicStateType.RIGIDBODY,
        #     scale=0.8,
        # ),
    ]
    traj_filepath = "metasim/cfg/tasks/binpick/bin_picking_v2.json"
    # traj_filepath = None
    # checker = JointPosChecker(
    #     obj_name="box_base",
    #     joint_name="box_joint",
    #     mode="le",
    #     radian_threshold=-14 / 180 * math.pi,
    # )

    reward_functions = []
    reward_weights = []
    reward_functions.append(obj_hand_distance)
    reward_weights.append(1.0)
    reward_functions.append(obj_height)
    reward_weights.append(1.0)
    # reward_functions.append(hand_orientation)
    # reward_weights.append(1.0)
    # reward_functions.append(hand_target)
    # reward_weights.append(0.0)
    reward_functions.append(obj_position)
    reward_weights.append(1.0)
    reward_functions.append(finger_width)
    reward_weights.append(1.0)
    # reward_functions.append(fine_obj_hand_distance)
    # reward_weights.append(0.0)
    # reward_functions.append(x_distance)
    # reward_weights.append(0.0)
    # reward_functions.append(fingerip_obj_distance)
    # reward_weights.append(0.0)
    reward_functions.append(obj_in_hand)
    reward_weights.append(1.0)
    # reward_functions.append(tiny_ee_vel)
    # reward_weights.append(0.0)

    # reward_functions.append(reward_joint_angle_0)
    # reward_weights.append(1.0)
    # reward_functions.append(reward_joint_angle_2)
    # reward_weights.append(1.0)
    # reward_functions.append(reward_joint_angle_3)
    # reward_weights.append(0.0)
    # reward_functions.append(reward_joint_angle_4)
    # reward_weights.append(1.0)
    # reward_functions.append(reward_joint_angle_6)
    # reward_weights.append(1.0)

    # reward_functions.append(naive_obj_hand_distance)
    # reward_weights.append(1.0)


    # reward_functions = [obj_hand_distance, obj_height, hand_orientation, hand_target, obj_position, finger_width, fine_obj_hand_distance]
    # reward_weights = [2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]




# @configclass
# class FetchCloseBoxCfg(BaseTaskCfg):
#     episode_length = 250
#     objects = [
#         ArticulationObjCfg(
#             name="box_base",
#             usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
#             urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
#             mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
#         ),
#     ]
#     traj_filepath = "metasim/cfg/tasks/fetch/fetch_example_v2.json"
#     checker = JointPosChecker(
#         obj_name="box_base",
#         joint_name="box_joint",
#         mode="le",
#         radian_threshold=-14 / 180 * math.pi,
#     )
