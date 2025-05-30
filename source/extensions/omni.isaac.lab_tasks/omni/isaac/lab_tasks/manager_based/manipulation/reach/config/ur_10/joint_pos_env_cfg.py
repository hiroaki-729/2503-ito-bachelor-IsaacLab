# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UR10_CFG  # isort: skip


##
# Environment configuration
##



@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10f
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override reward  
        self.rewards.handvelocity.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.hight.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.side_vel.params["asset_cfg"].body_names = ["ee_link"]  
        # self.rewards.position_and_velocity.params["asset_cfg"].body_names = ["ee_link"] 
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"] # 手先位置によって報酬決定(ee_link:手先位置)

        self.terminations.judge_hit.params["asset_cfg"].body_names = ["ee_link"]
        
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


# @configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption =False
