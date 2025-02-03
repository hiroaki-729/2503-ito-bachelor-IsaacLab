# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject,Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
import numpy as np
import math
def handvelocity(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,posreq=0.02,velreq=-2.0) -> torch.Tensor:
    # velreq=-3.2
    asset: RigidObject = env.scene[asset_cfg.name]  # どの報酬関数でもここは同じ
    command = env.command_manager.get_command(command_name)   # 7列の配列
    des_pos_b = command[:, :3]               # commandの最初の3列を切り取り
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)  ## 目標位置の座標
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore       # 手先位置の座標
    elbow_state=asset.data.body_state_w[:, 3, :3]
    wrist_state=asset.data.body_state_w[:, 4, :3]
    hand=curr_pos_w.to('cpu').detach().numpy().copy()
    elbow=elbow_state.to('cpu').detach().numpy().copy()
    wrist=wrist_state.to('cpu').detach().numpy().copy()
    # print(asset.data.body_state_w[:,[1,2], :3] )
    # print("sssssssssssssssssss",hand)
    # with open('/home2/isaac_env/hand.csv', 'a' , encoding= 'utf-8' ) as f: #手先位置をcsvファイルに格納
    #     for row in hand:
    #         f.write(",".join(f"{val:.4e}" for val in row) + "\n")
    # with open('/home2/isaac_env/elbow.csv', 'a' , encoding= 'utf-8' ) as f: #肘位置をcsvファイルに格納
    #         for row in elbow:
    #             f.write(",".join(f"{val:.4e}" for val in row) + "\n")
    # with open('/home2/isaac_env/wrist.csv', 'a' , encoding= 'utf-8' ) as f: #手首位置をcsvファイルに格納
    #         for row in wrist:
    #             f.write(",".join(f"{val:.4e}" for val in row) + "\n")
    pos_h=curr_pos_w[:,2]   # 手先の高さ
    # print(pos_h)
    h=pos_h.to('cpu').detach().numpy().copy()   # numpyに変換
    # with open('/home2/isaac_env/h.csv', 'a' , encoding= 'utf-8' ) as f:  # 手先高さをcsvファイルに格納
    #     print(h[0],file=f)
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)        # 手先と目標の距離
    judge_pos=torch.signbit(distance-posreq)                 # distanceがposreq以下かどうかの判定。真なら1、偽なら0を返す。
    judge_h=torch.signbit(pos_h-posreq) 
    # judge_pos=torch.signbit(curr_pos_w[:,2]-posreq) 
    vel=asset.data.body_vel_w [:, asset_cfg.body_ids[0], :3]            # 手先速度
    # print(vel[:,2])
    handvel=torch.abs(vel[:,2]-velreq)          # 手先の鉛直方向速度誤差
    v=vel[:,2].to('cpu').detach().numpy().copy() # numpyに変換
    # with open('/home2/isaac_env/vel.csv', 'a' , encoding= 'utf-8' ) as f: #手先速度をcsvファイルに格納
    #     print(v[0],file=f)
    # mu = 0.0     # 平均
    sigmavel =0.35 # 標準偏差
    # sigma =0.3  # 標準偏差
    norm=torch.exp(-handvel*handvel/(2*sigmavel*sigmavel))
    # normal_dist = torch.distributions.Normal(mu, sigmavel)
    # gauvel=normal_dist.log_prob(norm).exp()
    h_req=norm* judge_h.double()
    printr=h_req.double().to('cpu').detach().numpy().copy()
    pr=printr.max()*env.step_dt*0.5
    # with open('/home2/isaac_env/r_handvel.csv', 'a' , encoding= 'utf-8' ) as f:  # タイムステップにおける最大報酬
    #         print(pr,file=f)
    return norm* judge_h.double()
    # return pos_h

# 手先の運動軌道を正弦波で与える
def sin(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,width=0.2,period=1.0) -> torch.Tensor:
    # velreq=-3.2
    asset: RigidObject = env.scene[asset_cfg.name]  # どの報酬関数でもここは同じ
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore       # 手先位置の座標
    pos_h=curr_pos_w[:,2]   # 手先の高さ
    # print(pos_h)
    timestep=env.common_step_counter # タイムステップ(整数)
    maxtimestep=env.max_episode_length # 1エピソードにおける最大タイムステップ
    target=width*math.sin(period*(timestep/maxtimestep)*2*math.pi-math.pi/2)+width+0.2
    error=pos_h-target
    sigmapos =0.2 # 標準偏差
    reward=torch.exp(-error*error/(2*sigmapos*sigmapos))
    # with open('/home2/isaac_env/sin.csv', 'a' , encoding= 'utf-8' ) as f: #目標の手先高さをcsvファイルに格納
    #     print(target,file=f)
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
    # print(asset_cfg.body_ids)
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
