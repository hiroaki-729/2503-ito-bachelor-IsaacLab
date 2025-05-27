
# 伊藤弘顕がIsaacLabを改良したリポジトリ

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

## IsaacLabについて
強化学習を使って様々なロボットの運動を学習させる。特徴としてGPUを用いた並列計算が可能である。
## 使用した環境
以下のロボットur10の手先をターゲット位置に移動させる運動を学習させる環境を使用した(Isaac-Reach-UR10-v0)。    
(https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)   
![alt text](ur10.png)
## 学習させる運動タスク
ur10を用いて、初期姿勢より下にある水平面を叩く運動タスク。叩く強さ(打面に手先が到達する直前の手先速度)に応じた運動の変化を調べる。
## ファイル構成
|自分が編集した部分|ディレクトリ|
|----|---------|
|消費エネルギーを最小にする報酬(energy_consumption) | https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/rewards.py|
|エピソード終了条件設定(judge_hit)|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/terminations.py|
|ロボットの初期姿勢設定、ロボットのファイル(usd)のパス指定|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/universal_robots.py|
|ppoのパラメータ、学習時間指定|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/reach/config/ur_10/agents/skrl_ppo_cfg.yaml|
|ur10の学習環境|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/reach/config/ur_10/joint_pos_env_cfg.py|　
|平面を目標速度で叩くための報酬(handvelocity)|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/reach/mdp/rewards.py|
|学習環境の詳細(報酬の重みなど)|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/reach/reach_env_cfg.py|
||| 



|大事な情報|ディレクトリ|
|----|---------|
|初期状態指定する関数(reset_joints_by_scale) |https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/events.py|
|各エージェントが得た報酬(reward_buf)|https://github.com/hiroaki-729/2503-ito-bachelor-IsaacLab/blob/master/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/manager_based_rl_env.py|
## 引用先
引用したIsaacLabの情報は以下の通り。
### URL
https://github.com/isaac-sim/IsaacLab
### License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

### Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
