
from __future__ import annotations
import math, random, numpy as np
from typing import List, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from simulator import Simulator
from bots import load_all_bots, get_bot_registry
from utils.obs_builder import EgoObsBuilder

def wrap_with_sensor_noise(controllers, rng, pos_sigma=0.02, angle_sigma_rad=math.radians(2.0)):
    wrapped=[]
    for ctrl in controllers:
        if not hasattr(ctrl, 'get_controls'): wrapped.append(ctrl); continue
        _orig=ctrl.get_controls
        def noisy_get_controls(bot_coords, green_coords, red_coords, _o=_orig, _r=rng, _ps=pos_sigma, _as=angle_sigma_rad):
            def jxy(p): return (p[0]+_r.gauss(0.0,_ps), p[1]+_r.gauss(0.0,_ps))
            n_bot=[(jxy(p), ang + _r.gauss(0.0,_as)) for (p,ang) in bot_coords]
            n_green=[jxy(p) for p in green_coords]; n_red=[jxy(p) for p in red_coords]
            return _o(n_bot, n_green, n_red)
        ctrl.get_controls = noisy_get_controls; wrapped.append(ctrl)
    return wrapped

class RLProxyController:
    def __init__(self, index:int, builder:EgoObsBuilder, reward_cfg:Dict[str,float]|None=None):
        self.i=index; self.builder=builder; self.action=np.zeros(2,dtype=np.float32)
        self.last_phi=None; self.rew_acc=0.0; self.last_obs=None; self.t=0.0
        self.reward_cfg=dict(win_bonus=1.0, phi_scale=0.05)
        if reward_cfg: self.reward_cfg.update(reward_cfg)
    def set_action(self, a): self.action=np.array(a,dtype=np.float32).clip(-1.0,1.0)
    def get_and_clear_reward(self): r=self.rew_acc; self.rew_acc=0.0; return r
    def get_obs(self): return self.last_obs
    def _phi(self, red_coords, base_own, base_opp):
        s=0.0
        for r in red_coords:
            d_own=math.hypot(r[0]-base_own[0], r[1]-base_own[1]); d_opp=math.hypot(r[0]-base_opp[0], r[1]-base_opp[1])
            s += (d_own - d_opp)
        return s
    def get_controls(self, bot_coords, green_coords, red_coords):
        try:
            from bots.utility_functions import get_base_coords, get_opponent_index
            base_own=get_base_coords(self.i); base_opp=get_base_coords(get_opponent_index(self.i))
        except Exception:
            S=self.builder.field_size; base_own, base_opp=(0.0,0.0),(S,S)
        self.t += self.builder.default_dt
        obs=self.builder.build(self.i, bot_coords, red_coords, green_coords, base_own, base_opp, time_now=self.t); self.last_obs=obs
        phi_now=self._phi(red_coords, base_own, base_opp)
        if self.last_phi is not None: self.rew_acc += self.reward_cfg['phi_scale'] * (self.last_phi - phi_now)
        self.last_phi=phi_now
        return float(self.action[0]), float(self.action[1])

class FloorballEnv(gym.Env):
    metadata={"render_modes":["human"], "render_fps":20}
    def __init__(self, agent_side:str="random", teammate:str="SimpleBot2", opponents:List[str]|None=None,
                 frame_skip:int=2, sensor_pos_sigma:float=0.02, sensor_angle_sigma_deg:float=2.0,
                 physics_noise:Dict[str,float]|None=None, reward_cfg:Dict[str,float]|None=None,
                 builder:EgoObsBuilder|None=None, seed:int|None=None):
        super().__init__()
        load_all_bots(); self.REG=get_bot_registry()
        self.agent_side=agent_side; self.teammate_name=teammate; self.opponents=opponents or ["SimpleBot2","SimpleBot2"]
        if len(self.opponents)==1: self.opponents=[self.opponents[0], self.opponents[0]]
        self.frame_skip=max(1,int(frame_skip)); self.sensor_pos_sigma=float(sensor_pos_sigma); self.sensor_angle_sigma=math.radians(sensor_angle_sigma_deg)
        self.physics_noise=physics_noise or {"core_radius":0.02,"core_density":0.15,"core_friction":0.25,"core_restitution":0.20,"core_linear_damping":0.10,"core_angular_damping":0.20,"robot_density":0.10,"robot_friction":0.25,"robot_ang_damp":0.10,"robot_speed_scale":0.05}
        self.reward_cfg=reward_cfg or {"win_bonus":1.0, "phi_scale":0.05}
        self.builder=builder or EgoObsBuilder(); self._rng=random.Random(seed)
        self.action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.sim=None; self.proxy=None; self._last_info={}
    def _build_controllers(self, agent_team:str):
        agent_idx=0 if agent_team=="A" else 2; proxy=RLProxyController(agent_idx, self.builder, reward_cfg=self.reward_cfg)
        def make(name, idx): 
            if name not in self.REG: raise KeyError(f"Unknown bot: {name}"); 
            return self.REG[name](idx)
        if agent_team=="A":
            controllers=[proxy, self.REG[self.teammate_name](1), self.REG[self.opponents[0]](2), self.REG[self.opponents[1]](3)]
        else:
            controllers=[self.REG[self.opponents[0]](0), self.REG[self.opponents[1]](1), proxy, self.REG[self.teammate_name](3)]
        return controllers, proxy
    def reset(self, *, seed:int|None=None, options:Dict[str,Any]|None=None):
        if seed is not None: self._rng.seed(seed)
        agent_team = "A" if (self.agent_side!="random" and self.agent_side.upper().startswith("A")) else ("B" if self.agent_side!="random" else ("A" if self._rng.random()<0.5 else "B"))
        controllers, proxy = self._build_controllers(agent_team)
        controllers = wrap_with_sensor_noise(controllers, self._rng, pos_sigma=self.sensor_pos_sigma, angle_sigma_rad=self.sensor_angle_sigma)
        self.sim=Simulator(); sim_seed=self._rng.randrange(1_000_000_000); self.sim.init(controllers, True, noise=self.physics_noise, seed=sim_seed)
        self.proxy=proxy; self._last_info={"agent_team":agent_team, "sim_seed":sim_seed}
        obs=self.proxy.get_obs()
        if obs is None: self.sim.update(); obs=self.proxy.get_obs()
        flat=obs["flat"].astype(np.float32); self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)
        return flat, dict(self._last_info)
    def step(self, action: np.ndarray):
        assert self.sim is not None and self.proxy is not None, "Call reset() first"
        self.proxy.set_action(action); total_r=0.0; term=False; trunc=False
        for _ in range(self.frame_skip):
            self.sim.update(); total_r += self.proxy.get_and_clear_reward()
            if self.sim.is_game_over(): term=True; break
        obs=self.proxy.get_obs()["flat"].astype(np.float32); info=dict(self._last_info)
        if term:
            winner=self.sim.get_winner(); agent_team=self._last_info["agent_team"]
            won = ((winner==1 and agent_team=="A") or (winner==2 and agent_team=="B"))
            if won: total_r += self.reward_cfg.get("win_bonus",1.0)
            elif winner in (1,2): total_r -= self.reward_cfg.get("win_bonus",1.0)
            info["winner"]=winner
        return obs, float(total_r), term, trunc, info
    def render(self): pass
    def close(self): self.sim=None; self.proxy=None
