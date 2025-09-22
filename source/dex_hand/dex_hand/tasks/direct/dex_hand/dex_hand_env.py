# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .dex_hand_env_cfg import DexHandEnvCfg


class DexHandEnv(DirectRLEnv):
    cfg: DexHandEnvCfg

    def __init__(self, cfg: DexHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 获取关节索引
        self._active_joint_idx, _ = self.robot.find_joints(self.cfg.active_joints)
        self._target_joint_idx, _ = self.robot.find_joints([self.cfg.target_joint])
        
        # 用于追踪里程碑
        self.prev_target_angle = torch.zeros(self.num_envs, device=self.device)
        self.crossed_threshold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot.is_fixed_base = True  # 固定基座
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 仅应用到主动关节
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale, 
            joint_ids=self._active_joint_idx
        )

    def _get_observations(self) -> dict:
        # 观察值：2个主动关节位置 + 目标关节位置 + 2个主动关节速度
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self._active_joint_idx],  # 主动关节位置
                self.robot.data.joint_pos[:, self._target_joint_idx],  # 目标关节位置
                self.robot.data.joint_vel[:, self._active_joint_idx],  # 主动关节速度
            ),
            dim=-1, # 沿列拼接
        )
        observations =  {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        target_angle = self.robot.data.joint_pos[:, self._target_joint_idx].squeeze(-1)
        target_vel = torch.abs(self.robot.data.joint_vel[:, self._target_joint_idx].squeeze(-1))
        
        
        # 动态计算奖励
        
        rewards = compute_rewards(
            rew_scale_target_angle=self.cfg.rew_scale_target_angle,
            rew_scale_energy=self.cfg.rew_scale_energy,
            rew_scale_crossing_speed=self.cfg.rew_scale_crossing_speed,
            joint_angle=target_angle,
            joint_vel=target_vel,
            actions=self.actions,
        )
        
        # 检查是否首次越过分岔平面
        just_crossed = (self.prev_target_angle > 0) & (target_angle <= 0)
        self.crossed_threshold = self.crossed_threshold | just_crossed
        rewards += self.cfg.rew_scale_milestone * just_crossed.float()
        
        # 检查是否达到目标范围
        target_min = self.cfg.target_angle_range[0]
        target_max = self.cfg.target_angle_range[1]
        in_target_range = (target_angle >= target_min) & (target_angle <= target_max)
        rewards += self.cfg.rew_scale_success * in_target_range.float()
        
        self.prev_target_angle = target_angle.clone()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. 超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 2. 成功：在目标范围内保持稳定
        target_angle = self.robot.data.joint_pos[:, self._target_joint_idx].squeeze(-1)
        target_min = self.cfg.target_angle_range[0]
        target_max = self.cfg.target_angle_range[1]
        in_target_range = (target_angle >= target_min) & (target_angle <= target_max)
        stable = (torch.abs(self.robot.data.joint_vel[:, self._target_joint_idx]) < 0.1).squeeze(-1)
        success = in_target_range & stable
        
        # 3. 失败：超出关节限制
        failed = torch.zeros_like(time_out)
        for joint, limits in self.cfg.joint_limits.items():
            j_idx, _ = self.robot.find_joints([joint])
            # if j_idx.nelement() > 0:  # 确保找到了关节
            joint_pos = self.robot.data.joint_pos[:, j_idx].squeeze(-1)
            out_of_bounds = (joint_pos < limits[0]) | (joint_pos > limits[1])
            failed = failed | out_of_bounds
        
        # 成功或失败都会导致环境重置
        done = success | failed | time_out
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.data.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.robot.data.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]

        self.prev_target_angle = torch.zeros(self.num_envs, device=self.device)
        self.crossed_threshold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.robot.write_root_state_to_sim(
            default_root_state, env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            self.robot.data.joint_pos[env_ids],
            self.robot.data.joint_vel[env_ids],
            env_ids=env_ids,
        )



@torch.jit.script
def compute_rewards(
    rew_scale_target_angle: float,
    rew_scale_energy: float,
    rew_scale_crossing_speed: float,
    joint_angle: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    
    # 定义关键角度
    bifurcation_angle = 0.0  # 0度平面
    target_center = math.radians(-30.0)  # 目标中心角度 -30度
    target_min = math.radians(-40.0)     # 目标区间下限
    target_max = math.radians(-20.0)     # 目标区间上限
    
    # 定义穿越区域（0度平面附近的窄带）
    crossing_band_min = math.radians(-5.0)
    crossing_band_max = math.radians(5.0)
    in_crossing_band = (joint_angle >= crossing_band_min) & (joint_angle <= crossing_band_max)
    
    # 根据角度位置划分区域
    positive_region = joint_angle > bifurcation_angle
    negative_region = joint_angle <= bifurcation_angle
    target_region = (joint_angle >= target_min) & (joint_angle <= target_max)
    
    # 初始化奖励
    target_reward = torch.zeros_like(joint_angle)
    
    # 1. 正角度区域：鼓励减小角度（靠近0度平面）
    target_reward[positive_region] = rew_scale_target_angle * (
        bifurcation_angle - joint_angle[positive_region]
    )
    
    # 2. 负角度区域：鼓励达到目标区间
    dist_to_center = torch.abs(joint_angle - target_center)
    target_reward[negative_region] = rew_scale_target_angle * (
        1.0 - dist_to_center[negative_region] / (math.radians(20.0))
    )
    
    # 3. 目标区域内：确保有最小奖励
    min_in_target_reward = rew_scale_target_angle * 0.8
    target_reward[target_region] = torch.max(
        target_reward[target_region], 
        torch.ones_like(target_reward[target_region]) * min_in_target_reward
    )
    
    # 4. 穿越带奖励：鼓励快速穿越0度平面
    # 在穿越带内，速度越快（负速度）奖励越高
    crossing_speed_reward = torch.zeros_like(joint_angle)
    
    # 计算穿越速度分量（负速度表示向下穿越）
    crossing_speed_component = torch.clamp(-joint_vel, min=0)
    
    # 在穿越带内应用速度奖励
    crossing_speed_reward[in_crossing_band] = rew_scale_crossing_speed * crossing_speed_component[in_crossing_band]
    
    # 5. 穿越后奖励：鼓励保持穿越状态
    # 如果已经穿越到负角度区域，给予额外奖励
    crossed_reward = torch.zeros_like(joint_angle)
    crossed_reward[negative_region] = rew_scale_crossing_speed * 0.5
    
    # 能耗惩罚
    energy_penalty = rew_scale_energy * torch.sum(actions ** 2, dim=1)
    
    # 稳定性奖励
    stability_reward = torch.zeros_like(joint_angle)
    
    # 总奖励 = 位置奖励 + 穿越速度奖励 + 穿越后奖励 + 稳定性奖励 - 能耗惩罚
    total_reward = target_reward + crossing_speed_reward + crossed_reward + stability_reward - energy_penalty
    
    return total_reward
