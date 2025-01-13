# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(

        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        # usd_path=f"/home2/isaac-env/Collected_ur10_instanceable/ur10_instanceable.usd",
        usd_path=f"/home2/isaac-env/isaaclab/lib/python3.10/site-packages/isaacsim/extscache/omni.importer.urdf-1.14.1+106.0.0.lx64.r.cp310/data/urdf/robots/ur10//urdf/ur10/ur10.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            # max_linear_velocity=0,
        ),
        activate_contact_sensors=False,
    ),
    

    # spawn=sim_utils.UrdfFileCfg(
    #     asset_path=f"/home2/isaac-env/isaaclab/lib/python3.10/site-packages/isaacsim/exts/omni.isaac.motion_generation/motion_policy_configs/universal_robots/ur10/ur10_robot_suction.urdf",
    #     # asset_path=f"/home2/isaac-env/Collected_ur10_instanceable/ur10_instanceable.usd",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         max_depenetration_velocity=5.0,
            
    #     ),
    #     activate_contact_sensors=False,
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 関節角度の初期位置設定
        joint_pos={
            # "shoulder_pan_joint": 0.0,
            # "shoulder_lift_joint": -1.712,
            # "elbow_joint": 1.712,
            # "wrist_1_joint": 0.0,

            "shoulder_pan_joint": 0.0,
            # "shoulder_lift_joint": -3.141592/12,
            "shoulder_lift_joint": -3.141592/18,
            # "shoulder_pan_joint": -3.141592/12,
            # "shoulder_lift_joint": 0,
            "elbow_joint": 0.0,
            "wrist_1_joint": -3.141592/2,


            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0/10,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
