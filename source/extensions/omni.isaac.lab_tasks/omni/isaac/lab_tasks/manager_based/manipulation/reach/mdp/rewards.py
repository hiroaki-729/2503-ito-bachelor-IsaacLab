# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# ## 手先の速度における報酬
def handvelocity(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,posreq=0.1,velreq=0.1) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]  # どの報酬関数でもここは同じ
    command = env.command_manager.get_command(command_name)   # 7列の配列
    des_pos_b = command[:, :3]               # commandの最初の3列を切り取り
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)  ## 目標位置の座標
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore       # 手先位置の座標
    # print("zzzzzzzzzzzzzzzzzz",curr_pos_w[:,2])
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)        # 手先と目標の距離
    judge_pos=torch.signbit(distance-posreq)                 # distanceがposreq以下かどうかの判定。真なら1、偽なら0を返す。
    # judge_pos=torch.signbit(curr_pos_w[:,2]-posreq) 
    vel=asset.data.body_vel_w [:, asset_cfg.body_ids[0], :3]            # 手先速度
    handvel=torch.abs(vel[:,1]-velreq)          # 手先の鉛直方向速度誤差
    norm=handvel*handvel                     # 2乗誤差
    
    mu = 0.0     # 平均
    sigma = 1.0  # 標準偏差
    normal_dist = torch.distributions.Normal(mu, sigma)
    gau=normal_dist.log_prob(norm).exp()

    # print("aaaaaaaaaaaaaaaaaaaaaaaaaa",gau* judge_pos.float())
    return gau* judge_pos.float()



# # ある一定の速度で叩く
def position_and_velocity(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,req=0.7,reqrange=0.1,posrange=0.1) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]  # どの報酬関数でもここは同じ
    command = env.command_manager.get_command(command_name)   # 7列の配列
    # obtain the desired and current positions
    des_pos_b = command[:, :3]               # commandの最初の3列を切り取り
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)  ## 目標位置の座標
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore       # 手先位置の座標
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)        # 手先と目標の距離
    # require=torch.ones(env.scene.num_envs,device='cuda:0')*req                       # 手先と目標との許容範囲
    vel=asset.data.body_vel_w [:, asset_cfg.body_ids[0], :3]            # 手先速度
    spe=torch.norm(vel,dim=1)                   # 手先速さ
    judge_vel=torch.signbit(-reqrange-req+spe)                         # 速さが一定かどうか
    judge_pos=torch.signbit(-posrange+distance)
    reward=judge_pos.float()*judge_vel.float()
    # vel=asset.data.body_vel_w [:, asset_cfg.body_ids[0], :3]            # 手先速度
    return reward

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]  # どの報酬関数でもここは同じ
    command = env.command_manager.get_command(command_name)   # 7列の配列
    # obtain the desired and current positions
    des_pos_b = command[:, :3]               # commandの最初の3列を切り取り
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)  ## 目標位置の座標
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore       # 手先位置の座標
    # joint_limit=asset.data.soft_joint_pos_limits

    joint_limit=asset.data.joint_pos
    # print(joint_limit)
    return torch.norm(curr_pos_w - des_pos_w, dim=1)



def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)
