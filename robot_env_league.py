
#!/usr/bin/env python3
"""
robot_env_league.py (with shaping)
Adds:
- Ball-shaping potential PBRS: R_balls = gamma * Phi_balls(s') - Phi_balls(s)
  where Phi_balls = mean_g w(dist_to_opp_goal) - mean_r w(dist_to_opp_goal).
  No ball pairing required.
- Movement/anti-stuck tick: every M steps, give +m if cumulative displacement >= thresh, else -m.
"""
from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Callable, Optional, Dict, Any

from simulator import Simulator
from utils.obs_builder import EgoObsBuilder

class RobotGameEnvLeague(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_idx: int = 0,
        controller_factory: Optional[Callable[[int], list]] = None,
        field_size: float = 1.5,
        k_red: int = 8,
        k_green: int = 4,
        goal_reward: float = 10.0,
        red_reward: float = 0.0,
        goal_radius: float = 0.10,
        randomize: bool = True,
        physics_noise: Optional[Dict[str, float]] = None,
        max_steps: int = 1200,
        seed: Optional[int] = None,
        # --- New shaping knobs ---
        gamma: float = 0.99,                # discount used for PBRS (match PPO gamma)
        use_ball_shaping: bool = True,
        ball_kernel: str = "inv",           # "inv" or "exp"
        ball_sigma: float = 0.25,           # length-scale for exp kernel
        ball_eps: float = 1e-3,             # epsilon for inverse kernel
        ball_weight: float = 0.5,           # multiplier for R_balls
        move_interval: int = 20,            # steps between movement checks
        move_thresh: float = 0.05,          # meters of cumulative displacement
        move_reward: float = 0.02           # reward magnitude at movement tick
    ):
        super().__init__()
        self.agent_idx = agent_idx
        self.controller_factory = controller_factory
        self.field_size = field_size
        self.goal_reward = float(goal_reward)
        self.red_reward = float(red_reward)
        self.goal_radius = float(goal_radius)
        self.randomize = bool(randomize)
        self.physics_noise = physics_noise or {}
        self.max_steps = int(max_steps)

        # Shaping params
        self.gamma = float(gamma)
        self.use_ball_shaping = bool(use_ball_shaping)
        self.ball_kernel = str(ball_kernel)
        self.ball_sigma = float(ball_sigma)
        self.ball_eps = float(ball_eps)
        self.ball_weight = float(ball_weight)
        self.move_interval = int(move_interval)
        self.move_thresh = float(move_thresh)
        self.move_reward = float(move_reward)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obs_builder = EgoObsBuilder(field_size=field_size, k_red=k_red, k_green=k_green)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_builder.flat_dim,), dtype=np.float32)

        self.sim = Simulator()
        self._t = 0
        self._scores_prev = None
        self._phi_prev = 0.0
        self._move_tick = 0
        self._move_cum = 0.0
        self._pos_prev = None

    # ---------- helpers ----------
    def _read_scores(self):
        scores = getattr(self.sim, "scores", None)
        reds = getattr(self.sim, "red_core_counts", None)
        return scores, reds

    def _team_keys(self):
        own_team = 0 if self.agent_idx in (0,1) else 1
        opp_team = 1 - own_team
        return own_team, opp_team

    def _opp_goal(self):
        return (self.field_size, self.field_size) if self.agent_idx in (0,1) else (0.0, 0.0)

    def _w(self, d: float) -> float:
        if self.ball_kernel == "exp":
            return math.exp(-d / max(1e-9, self.ball_sigma))
        # default inverse
        return 1.0 / (d + self.ball_eps)

    def _phi_balls(self) -> float:
        opp_goal = self._opp_goal()
        gx = [tuple(c.position) for c in self.sim.green_cores]
        rx = [tuple(c.position) for c in self.sim.red_cores]
        def score(balls):
            if not balls: return 0.0
            s = 0.0
            for (x,y) in balls:
                dx = x - opp_goal[0]; dy = y - opp_goal[1]
                s += self._w(math.hypot(dx, dy))
            return s / float(len(balls))
        return score(gx) - score(rx)

    # ---------- Gym API ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self.controller_factory is None:
            raise RuntimeError("RobotGameEnvLeague requires a controller_factory to provide controllers per episode.")
        controllers = list(self.controller_factory(self.agent_idx))
        assert len(controllers) == 4, "controller_factory must return a list of 4 controllers"
        controllers[self.agent_idx] = self  # env handles agent controls
        for i,c in enumerate(controllers):
            if c is None:
                raise RuntimeError(f"Controller at index {i} is None")
        self.sim.init(controllers, randomize=self.randomize, noise=self.physics_noise, seed=seed)
        self._t = 0
        self._scores_prev, _ = self._read_scores()
        # initialize shaping state
        self._phi_prev = self._phi_balls()
        self._move_tick = 0
        self._move_cum = 0.0
        self._pos_prev = np.array(self.sim.robots[self.agent_idx].position, dtype=np.float32)
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.next_action = action
        self.sim.update()
        self._t += 1

        obs = self._get_observation()
        terminated = bool(self.sim.is_game_over())
        truncated = self._t >= self.max_steps

        scores, reds = self._read_scores()
        own, opp = self._team_keys()

        delta_own = (scores[own] - self._scores_prev[own]) if self._scores_prev is not None else 0
        delta_opp = (scores[opp] - self._scores_prev[opp]) if self._scores_prev is not None else 0
        self._scores_prev = list(scores)

        # Base reward: score delta as before
        reward = self.goal_reward * (delta_own - delta_opp)

        # Ball-shaping (PBRS)
        if self.use_ball_shaping:
            phi_now = self._phi_balls()
            r_balls = self.gamma * phi_now - self._phi_prev
            self._phi_prev = phi_now
            reward += self.ball_weight * float(r_balls)

        # Movement/anti-stuck tick
        if self.move_interval > 0 and self.move_reward != 0.0:
            pos = np.array(self.sim.robots[self.agent_idx].position, dtype=np.float32)
            self._move_cum += float(np.linalg.norm(pos - self._pos_prev))
            self._pos_prev = pos
            self._move_tick += 1
            if self._move_tick >= self.move_interval:
                reward += (self.move_reward if (self._move_cum >= self.move_thresh) else -self.move_reward)
                self._move_tick = 0
                self._move_cum = 0.0

        info = {"score_own": scores[own], "score_opp": scores[opp],
                "red_counts": tuple(reds) if reds is not None else None}

        return obs, float(reward), terminated, truncated, info

    # Called by simulator for the agent slot
    def get_controls(self, bot_coords, green_coords, red_coords):
        a = getattr(self, "next_action", None)
        if a is None:
            return 0.0, 0.0
        return float(a[0]), float(a[1])

    def _get_observation(self):
        bot_coords = [(tuple(b.position), b.angle) for b in self.sim.robots]
        red_coords = [tuple(c.position) for c in self.sim.red_cores]
        green_coords = [tuple(c.position) for c in self.sim.green_cores]
        base_own = (0.0, 0.0) if self.agent_idx in (0,1) else (self.field_size, self.field_size)
        base_opp = (self.field_size, self.field_size) if self.agent_idx in (0,1) else (0.0, 0.0)
        obs_dict = self.obs_builder.build(self.agent_idx, bot_coords, red_coords, green_coords, base_own, base_opp, time_now=self._t*1.0)
        return obs_dict["flat"]
