"""Train PPO for reaching task."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rootutils
import torch
import tyro
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.utils import ObsSaver
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.kinematics_utils import get_curobo_models
from scipy.spatial.transform import Rotation as R
from curobo.types.math import Pose

def rpy_to_quat(rpy: ndarray) -> ndarray:
    """ conver rpy to quaternion in the order of zyx """
    r, p, y = rpy
    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)
    cp = np.cos(p * 0.5)
    sp = np.sin(p * 0.5)
    cr = np.cos(r * 0.5)
    sr = np.sin(r * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([x, y, z, w], dtype=np.float32)


def move_to_pose(
    current_robot_q, robot_ik, ee_pos_target, ee_quat_target
):
    """Move the robot to the target pose."""

    # ee_pos_target = torch.tensor(ee_pos_target, device=current_robot_q.device, dtype=torch.float32)
    # ee_quat_target = torch.tensor(ee_quat_target, device=current_robot_q.device, dtype=torch.float32)
    seed_config = current_robot_q.unsqueeze(0)
    # seed_config = seed_config[:, :].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
    seed_config = seed_config[:, :].unsqueeze(1).tile([1, 20, 1])

    # print("current_robot_q", current_robot_q)
    # result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)
    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target),
                                  num_seeds = 1000,
                                    seed_config=seed_config,
                                    )
    # print("result", result)
    ik_succ = result.success.squeeze(1)

    # print(current_robot_q.shape)
    # if failed
    if not ik_succ.any():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!! IK failed")
        joint_action = current_robot_q
    else:
        joint_action = result.solution[ik_succ, 0].clone()
        joint_action = joint_action.squeeze(0)

    # print("ik_succ", ik_succ)
    # q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    # q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
    # actions = [
    #     {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    # ]

    return joint_action


@dataclass
class Args(ScenarioCfg):
    """Arguments for training PPO."""

    task: str = "binpick:bin_picking"
    robot: str = "franka"
    num_envs: int = 8
    sim: Literal["isaaclab", "isaacgym", "pyrep", "pybullet", "sapien", "mujoco"] = "isaacgym"
    render: Literal["isaaclab", "isaacgym", "pyrep", "pybullet", "sapien", "mujoco"] = "isaacgym"
    save_dir: str = ""
    load_dir: str = ""



args = tyro.cli(Args)

# class MetaSimVecEnv(VectorEnv):
#     """Vectorized environment for MetaSim that supports parallel RL training."""

#     def __init__(
#         self,
#         scenario: ScenarioCfg | None = None,
#         sim: str = "isaacgym",
#         task_name: str | None = None,
#         num_envs: int | None = 4,
#     ):
#         print("scenario", scenario)
#         print("task_name", task_name)
#         print("num_envs", num_envs)
#         print("sim", sim)
#         input("press enter to continue")


#         """Initialize the environment."""
#         if scenario is None:
#             scenario = ScenarioCfg(task="pick_cube", robot="franka")
#             scenario.task = task_name
#             scenario.num_envs = num_envs
#             scenario = ScenarioCfg(**vars(scenario))
#         self.num_envs = scenario.num_envs
#         env_class = get_sim_env_class(SimType(sim))
#         env = env_class(scenario)
#         self.env: EnvWrapper[BaseSimHandler] = env
#         self.render_mode = None  # XXX
#         self.scenario = scenario

#         # Get candidate states
#         self.candidate_init_states, _, _ = get_traj(scenario.task, scenario.robot)
#         # self.candidate_init_states = [np.zeros((len(scenario.robot.joint_limits),))]
#         # num_joints = len(scenario.robot.joint_limits)
#         # default_joint_pos = np.zeros((num_joints,))

#         # self.candidate_init_states = [
#         #     {
#         #         "robots": {"joint_pos": default_joint_pos},
#         #         "objects": {}  # or include some default object state here if needed
#         #     }
#         # ]


#         # XXX: is the inf space ok?
#         self.single_observation_space = spaces.Box(-np.inf, np.inf)
#         self.single_action_space = spaces.Box(-np.inf, np.inf)

#     ############################################################
#     ## Gym-like interface
#     ############################################################
#     def reset(self, env_ids: list[int] | None = None, seed: int | None = None):
#         """Reset the environment."""
#         if env_ids is None:
#             env_ids = list(range(self.num_envs))
#         init_states = self.unwrapped._get_default_states(seed)
#         self.env.reset(states=init_states, env_ids=env_ids)
#         return self.unwrapped._get_obs(), {}

#     def step(self, actions: list[dict]):
#         """Step the environment."""
#         # joint space

#         # print(len(actions))
#         # print("actions", actions[0])

#         _, _, success, timeout, _ = self.env.step(actions)
#         obs = self.unwrapped._get_obs()

#         rewards = self.unwrapped._calculate_rewards()
#         return obs, rewards, success, timeout, {}

#         # # ee space
#         # """Move the robot to the target pose."""
#         # states = self.env.handler.get_states()
#         # curr_robot_q = states.robots[robot.name].joint_pos
#         # seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

#         # result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

#         # q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
#         # ik_succ = result.success.squeeze(1)
#         # q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
#         # q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
#         # actions = [
#         #     {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
#         # ]


#     def render(self):
#         """Render the environment."""
#         return self.env.render()

#     def close(self):
#         """Close the environment."""
#         self.env.close()

#     ############################################################
#     ## Helper methods
#     ############################################################
#     def _get_obs(self):
#         ## TODO: put this function into task definition?
#         ## TODO: use torch instead of numpy
#         """Get current observations for all environments."""
#         states = self.env.handler.get_states()
#         joint_pos = states.robots["franka"].joint_pos
#         panda_hand_index = states.robots["franka"].body_names.index("panda_hand")
#         ee_pos = states.robots["franka"].body_state[:, panda_hand_index, :7]
#         # get the object positions
#         obj_pos = states.objects["block_red"].root_state[:, :7]
#         obs = torch.cat([joint_pos, ee_pos, obj_pos], dim=1)
#         return obs

#     def _calculate_rewards(self):
#         """Calculate rewards based on distance to origin."""
#         states = self.env.handler.get_states()
#         tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)
#         # print("task reward functions", self.scenario.task.reward_functions)
#         for reward_fn, weight in zip(self.scenario.task.reward_functions, self.scenario.task.reward_weights):
#             tot_reward += weight * reward_fn(states, self.scenario.robot.name)
#         return tot_reward

#     def _get_default_states(self, seed: int | None = None):
#         """Generate default reset states."""
#         ## TODO: use non-reqeatable random choice when there is enough candidate states?
#         return random.Random(seed).choices(self.candidate_init_states, k=self.num_envs)




class MetaSimVecEEEnv(VectorEnv):
    """Vectorized environment for MetaSim that supports parallel RL training."""

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        sim: str = "isaacgym",
        task_name: str | None = None,
        num_envs: int | None = 4,
    ):
        print("scenario", scenario)
        print("task_name", task_name)
        print("num_envs", num_envs)
        print("sim", sim)
        input("press enter to continue")


        """Initialize the environment."""
        if scenario is None:
            scenario = ScenarioCfg(task="pick_cube", robot="franka")
            scenario.task = task_name
            scenario.num_envs = num_envs
            scenario = ScenarioCfg(**vars(scenario))
        self.num_envs = scenario.num_envs
        env_class = get_sim_env_class(SimType(sim))
        env = env_class(scenario)
        self.env: EnvWrapper[BaseSimHandler] = env
        self.render_mode = None  # XXX
        self.scenario = scenario

        # Get candidate states
        self.candidate_init_states, _, _ = get_traj(scenario.task, scenario.robot)
        # self.candidate_init_states = [np.zeros((len(scenario.robot.joint_limits),))]
        # num_joints = len(scenario.robot.joint_limits)
        # default_joint_pos = np.zeros((num_joints,))

        # self.candidate_init_states = [
        #     {
        #         "robots": {"joint_pos": default_joint_pos},
        #         "objects": {}  # or include some default object state here if needed
        #     }
        # ]


        # XXX: is the inf space ok?
        self.single_observation_space = spaces.Box(-np.inf, np.inf)
        self.single_action_space = spaces.Box(-np.inf, np.inf)


        robot = scenario.robot
        kin_model, do_fk,robot_ik = get_curobo_models(robot)
        self.robot_ik = robot_ik
        self.do_fk = do_fk
    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self, env_ids: list[int] | None = None, seed: int | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        init_states = self.unwrapped._get_default_states(seed)
        self.env.reset(states=init_states, env_ids=env_ids)
        return self.unwrapped._get_obs(), {}

    def step(self, actions: list[dict]):

        # set ee_target pos range
        ee_pos_target_range = [
            [0.5, 0.55],  # x
            [-0.2, 0.2],  # y
            [0.1, 0.8],  # z
        ]

        self.action_range = np.array([[-0.02, -0.01, -0.02, -0.02 ],[0.02, 0.01, 0.02, 0.04]])
        # print(self.action_range[0,:], self.action_range[1,:])
        """Step the environment."""
        obs = self.unwrapped._get_obs()
        joint_actions = []
        # joint_actions =
        for env_id in range(len(actions)):
            ee_action = actions[env_id]["ee_pose_target"]
            # print("policy ee_action", ee_action)
            # rescale the ee_action to the range
            ee_action = self.action_range[0,:] + 0.5*(self.action_range[1, :] - self.action_range[0, :]) * (ee_action+1.0)


            ee_action = torch.tensor(ee_action, device=obs.device, dtype=torch.float32)
            # print("ee_action", ee_action)
            ee_rpy_target = ee_action[3:6]
            # get the current end-effector position and quaternion
            # print("obs", obs[env_id])
            ee_pos_current = obs[env_id, 9:12]
            ee_quat_current = obs[env_id, 12:16]

            ee_quat_target = ee_quat_current.clone()
            ee_pos_target = ee_pos_current.clone()
            ee_pos_target = ee_pos_current + ee_action[:3]
            #clamp the ee_pos_target to the range
            ee_pos_target[0] = torch.clamp(ee_pos_target[0], ee_pos_target_range[0][0], ee_pos_target_range[0][1])
            ee_pos_target[1] = torch.clamp(ee_pos_target[1], ee_pos_target_range[1][0], ee_pos_target_range[1][1])
            ee_pos_target[2] = torch.clamp(ee_pos_target[2], ee_pos_target_range[2][0], ee_pos_target_range[2][1])

            # ee_pos_target = torch.tensor([0.69, 0.0, 0.46], device=obs.device, dtype=torch.float32)
            ee_quat_target = torch.tensor([0.0, 1.0, 0.0, 0.0], device=obs.device, dtype=torch.float32)

            # print("ee_pos_target", ee_pos_target)
            joint_action = move_to_pose(
                obs[env_id,:7], self.robot_ik, ee_pos_target, ee_quat_target
            )
            # add the gripper action to the joint action
            # the shape the of joint action is [7]
            joint_action = torch.cat([joint_action,
                                        torch.tensor([ee_action[3]], device=joint_action.device),
                                        torch.tensor([ee_action[3]], device=joint_action.device)
                                        ], dim=0)

            joint_actions.append(joint_action)


        action_dicts = [
            {"dof_pos_target": dict(zip(self.scenario.robot.joint_limits.keys(), action))} for action in joint_actions
        ]
        # print("action_dicts", action_dicts[0])

        _, _, success, timeout, _ = self.env.step(action_dicts)
        obs = self.unwrapped._get_obs()
        rewards = self.unwrapped._calculate_rewards()
        return obs, rewards, success, timeout, {}

        # # ee space
        # """Move the robot to the target pose."""
        # states = self.env.handler.get_states()
        # curr_robot_q = states.robots[robot.name].joint_pos
        # seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

        # result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

        # q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
        # ik_succ = result.success.squeeze(1)
        # q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
        # q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
        # actions = [
        #     {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
        # ]


    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Helper methods
    ############################################################
    def _get_obs(self):
        ## TODO: put this function into task definition?
        ## TODO: use torch instead of numpy
        """Get current observations for all environments."""
        states = self.env.handler.get_states()
        joint_pos = states.robots["franka"].joint_pos
        # joint_pos = joint_pos[:, -2:]
        panda_hand_index = states.robots["franka"].body_names.index("panda_hand")
        # ee_pos = states.robots["franka"].body_state[:, panda_hand_index, :7]
        ee_position, ee_quaternion = self.do_fk(states.robots["franka"].joint_pos)
        ee_pos = torch.cat([ee_position, ee_quaternion], dim=1)
        # get the object positions
        obj_pos = states.objects["block_red"].root_state[:, :7]
        obs = torch.cat([joint_pos, ee_pos, obj_pos], dim=1)
        # print("obs", obs[0])
        return obs

    def _calculate_rewards(self):
        """Calculate rewards based on distance to origin."""
        states = self.env.handler.get_states()
        tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)
        # print("task reward functions", self.scenario.task.reward_functions)
        for reward_fn, weight in zip(self.scenario.task.reward_functions, self.scenario.task.reward_weights):
            tot_reward += weight * reward_fn(states, self.scenario.robot.name)
        return tot_reward

    def _get_default_states(self, seed: int | None = None):
        """Generate default reset states."""
        ## TODO: use non-reqeatable random choice when there is enough candidate states?
        return random.Random(seed).choices(self.candidate_init_states, k=self.num_envs)



class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEEEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robot.joint_limits

        # TODO: customize action space?
        # self.action_space = spaces.Box(
        #     low=np.array([lim[0] for lim in joint_limits.values()]),
        #     high=np.array([lim[1] for lim in joint_limits.values()]),
        #     dtype=np.float32,
        # )


        # self.action_space = spaces.Box(
        #     low=np.array([-2.0, -0.001, -2.0, 1.0]),
        #     high=np.array([2.0, 0.001, 2.0, 4.0]),
        #     dtype=np.float32,
        # )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # TODO: customize observation space?
        # Observation space: joint positions + end effector position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits) + 14,),  # joints + XYZ
            dtype=np.float32,
        )
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(2+3+7,),  # joints + XYZ
        #     dtype=np.float32,
        # )

        self.env = env
        self.render_mode = None  # XXX
        super().__init__(self.env.num_envs, self.observation_space, self.action_space)

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        self.action_dicts = [
            {"dof_pos_target": dict(zip(self.env.scenario.robot.joint_limits.keys(), action))} for action in actions
        ]

        # define the ee_pos target keys
        # ee_pos_target_keys = ["ee_pos_x", "ee_pos_y", "ee_pos_z",
        #                         "ee_r", "ee_p", "ee_y",
        #                         "gripper_width"]
        self.action_dicts = [
            {"ee_pose_target": action} for action in actions
        ]


    def step_wait(self):
        """Wait for the step to complete."""
        obs, rewards, success, timeout, _ = self.env.step(self.action_dicts)

        dones = success | timeout
        if dones.any():
            self.env.reset(env_ids=dones.nonzero().squeeze(-1).tolist())

        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if dones[env_id]:
                extra[env_id]["terminal_observation"] = obs[env_id].cpu().numpy()
            extra[env_id]["TimeLimit.truncated"] = timeout[env_id].item() and not success[env_id].item()

        obs = self.env.unwrapped._get_obs()

        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), extra

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        """Get images from the environment."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set an attribute of the environment."""
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call a method of the environment."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if the environment is wrapped by a given wrapper class."""
        raise NotImplementedError


def train_ppo():
    """Train PPO for reaching task."""
    ## Choice 1: use scenario config to initialize the environment
    # scenario = ScenarioCfg(**vars(args))
    # scenario.cameras = []  # XXX: remove cameras to avoid rendering to speed up
    # metasim_env = MetaSimVecEEEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)

    ## Choice 2: use gym.make to initialize the environment
    # metasim_env = gym.make("reach_origin", num_envs=args.num_envs)



    # # check whether load a pre-trained model
    # if args.load_dir:
    #     model.policy.load_state_dict(PPO.load(args.load_dir).policy.state_dict())
    #     model.set_env(env)



    # Inference and Save Video
    # add cameras to the scenario
    args.num_envs = 2
    scenario = ScenarioCfg(**vars(args))
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    metasim_env = MetaSimVecEEEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)
    task_name = scenario.task.__class__.__name__[:-3]

    env = StableBaseline3VecEnv(metasim_env)

    # # PPO configuration
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=1e-3,
    #     ent_coef=0.01,
    #     n_steps=512,
    #     batch_size=64,
    #     n_epochs=20,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )



    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}.mp4")
    # load the model
    model = PPO.load(args.load_dir, env=env)

    # inference
    obs, _ = metasim_env.reset()
    obs_orin = metasim_env.env.handler.get_states()
    obs_saver.add(obs_orin)
    for _ in range(100):
        actions, _ = model.predict(obs.cpu().numpy(), deterministic=True)
        # action_dicts = [
        #     {"dof_pos_target": dict(zip(metasim_env.scenario.robot.joint_limits.keys(), action))} for action in actions
        # ]

        action_dicts = [
            {"ee_pose_target": action} for action in actions
        ]

        obs, _, _, _, _ = metasim_env.step(action_dicts)

        obs_orin = metasim_env.env.handler.get_states()
        obs_saver.add(obs_orin)
    obs_saver.save()





def main():
    train_ppo()

if __name__ == "__main__":
    main()
