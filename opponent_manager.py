
#!/usr/bin/env python3
"""
opponent_manager.py
- Creates controllers for teammate/opponents per episode:
  * Named bots from registry
  * PPO self-play opponents loaded from SB3 .zip (policy.predict)
- Call .sample_controllers(agent_idx) to get a list of 4 controllers.
"""
from __future__ import annotations
from typing import List, Callable, Optional, Dict
import random
import numpy as np

from bots import get_bot_registry, load_all_bots
from utils.obs_builder import EgoObsBuilder
import torch

class PPOPolicyController:
    """Wrap an SB3 PPO policy zip as a simulator-compatible controller."""
    def __init__(self, zip_path:str, agent_index:int, field_size:float=1.5, k_red:int=8, k_green:int=4, device:str="cpu"):
        from stable_baselines3 import PPO
        self.policy = PPO.load(zip_path, device=device, print_system_info=False)
        self.builder = EgoObsBuilder(field_size=field_size, k_red=k_red, k_green=k_green)
        self.i = agent_index
        self.device = device

    def _bases(self, S:float):
        return ((0.0,0.0),(S,S)) if self.i in (0,1) else ((S,S),(0.0,0.0))

    def get_controls(self, bot_coords, green_coords, red_coords):
        S = self.builder.field_size
        base_own, base_opp = self._bases(S)
        obs = self.builder.build(self.i, bot_coords, red_coords, green_coords, base_own, base_opp, time_now=0.0)["flat"]
        action, _ = self.policy.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        return float(a[0]), float(a[1])

class OpponentManager:
    def __init__(self, field_size:float=1.5, k_red:int=8, k_green:int=4, seed:int=0):
        load_all_bots()
        self.REG = get_bot_registry()
        self.field_size = field_size; self.k_red = k_red; self.k_green = k_green
        self.rng = random.Random(seed)
        self.fixed_bots : List[str] = []             # names in registry
        self.selfplay_pool : List[str] = []          # file paths to SB3 .zip checkpoints
        self.device = "cpu"

    def set_device(self, device:str):
        self.device = device

    def add_fixed(self, *names:str):
        for n in names:
            if n not in self.REG:
                raise ValueError(f"Unknown bot '{n}'. Available: {', '.join(sorted(self.REG.keys()))}")
            self.fixed_bots.append(n)

    def add_selfplay_zip(self, *paths:str):
        self.selfplay_pool.extend(paths)

    def register_snapshot(self, path:str):
        self.selfplay_pool.append(path)

    def _make_bot(self, name:str, idx:int):
        return self.REG[name](idx)

    def _make_selfplay(self, zip_path:str, idx:int):
        return PPOPolicyController(zip_path, idx, field_size=self.field_size, k_red=self.k_red, k_green=self.k_green, device=self.device)

    def sample_controllers(self, agent_idx:int):
        # teammate
        team_name = self.rng.choice(self.fixed_bots) if self.fixed_bots else "SimpleBot2"
        # opponents can be mix of fixed and selfplay
        opp_ctrls = []
        for _j in range(2):
            if self.selfplay_pool and self.rng.random() < 0.5:
                opp_ctrls.append(("selfplay", self.rng.choice(self.selfplay_pool)))
            else:
                opp_ctrls.append(("fixed", self.rng.choice(self.fixed_bots) if self.fixed_bots else "SimpleBot2"))

        # assemble in team slots
        if agent_idx in (0,1):
            c0 = None  # agent filled by env
            c1 = self._make_bot(team_name, 1 if agent_idx==0 else 0)
            c2 = self._make_selfplay(opp_ctrls[0][1], 2) if opp_ctrls[0][0]=="selfplay" else self._make_bot(opp_ctrls[0][1], 2)
            c3 = self._make_selfplay(opp_ctrls[1][1], 3) if opp_ctrls[1][0]=="selfplay" else self._make_bot(opp_ctrls[1][1], 3)
            return [c0, c1, c2, c3]
        else:
            c2 = None
            c3 = self._make_bot(team_name, 3 if agent_idx==2 else 2)
            c0 = self._make_selfplay(opp_ctrls[0][1], 0) if opp_ctrls[0][0]=="selfplay" else self._make_bot(opp_ctrls[0][1], 0)
            c1 = self._make_selfplay(opp_ctrls[1][1], 1) if opp_ctrls[1][0]=="selfplay" else self._make_bot(opp_ctrls[1][1], 1)
            return [c0, c1, c2, c3]
