
#!/usr/bin/env python3
from __future__ import annotations
import os, argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from robot_env_league import RobotGameEnvLeague
from opponent_manager import OpponentManager

def make_league_env(n_envs:int, opp_mgr:OpponentManager, args):
    def factory(rank:int):
        def _thunk():
            def controller_factory(agent_idx:int):
                ctrls = opp_mgr.sample_controllers(agent_idx)
                ctrls[agent_idx] = None
                return ctrls
            return RobotGameEnvLeague(
                agent_idx=0 if (rank % 2 == 0) else 2,
                controller_factory=controller_factory,
                field_size=args.field_size, k_red=args.k_red, k_green=args.k_green,
                goal_reward=args.goal_reward, red_reward=args.red_reward,
                goal_radius=args.goal_radius, randomize=not args.no_randomize,
                physics_noise=None, max_steps=args.max_steps, seed=args.seed + rank
            )
        return _thunk
    if n_envs > 1:
        return SubprocVecEnv([factory(i) for i in range(n_envs)])
    return DummyVecEnv([factory(0)])

class SnapshotCallback(BaseCallback):
    def __init__(self, save_freq:int, save_dir:str, opp_mgr:OpponentManager, verbose:int=1):
        super().__init__(verbose); self.save_freq=save_freq; self.save_dir=save_dir; self.opp_mgr=opp_mgr
        os.makedirs(save_dir, exist_ok=True)
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"ppo_snap_{self.n_calls}.zip")
            self.model.save(path)
            self.opp_mgr.register_snapshot(path)
            if self.verbose: print(f"[league] snapshot saved and registered: {path}")
        return True

class VecNormSaveCallback(BaseCallback):
    def __init__(self, vecnorm:VecNormalize, save_path:str, freq:int=50_000, verbose:int=1):
        super().__init__(verbose); self.vecnorm=vecnorm; self.save_path=save_path; self.freq=freq
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.vecnorm.save(self.save_path)
            if self.verbose: print(f"[vecnorm] saved -> {self.save_path}")
        return True

class CriticWarmupCallback(BaseCallback):
    def __init__(self, warmup_steps:int, verbose:int=1):
        super().__init__(verbose); self.warmup_steps=warmup_steps
        self.actor_frozen=False; self.actor_unfrozen=False
    def _freeze_actor(self, freeze:bool):
        pol = self.model.policy
        for p in pol.mlp_extractor.policy_net.parameters(): p.requires_grad = not (freeze)
        for p in pol.action_net.parameters():             p.requires_grad = not (freeze)
    def _on_training_start(self) -> None:
        if self.warmup_steps > 0 and not self.actor_frozen:
            self._freeze_actor(True); self.actor_frozen=True
            if self.verbose: print(f"[critic-warmup] Actor frozen for first {self.warmup_steps} steps")
    def _on_step(self) -> bool:
        if self.actor_frozen and not self.actor_unfrozen and (self.num_timesteps >= self.warmup_steps):
            self._freeze_actor(False); self.actor_unfrozen=True
            if self.verbose: print(f"[critic-warmup] Actor unfrozen at step {self.num_timesteps}")
        return True

class WinRateCallback(BaseCallback):
    def __init__(self, opp_mgr_fixed:OpponentManager, args, train_vecnorm:VecNormalize|None,
                 eval_freq:int=50_000, n_episodes:int=16, verbose:int=1):
        super().__init__(verbose)
        self.opp_mgr_fixed = opp_mgr_fixed
        self.args = args
        self.train_vecnorm = train_vecnorm
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        # Build single VecEnv evaluator
        eval_env = make_league_env(1, self.opp_mgr_fixed, self.args)
        if self.train_vecnorm is not None:
            eval_vec = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_vec.load_running_stats(self.train_vecnorm)
            env = eval_vec
        else:
            env = eval_env

        # VecEnv API: obs, rewards, dones, infos (4-tuple)
        obs = env.reset()
        wins = 0; ties = 0
        for _ in range(self.n_episodes):
            done = False
            # For VecEnv, obs is batch of size 1
            if isinstance(obs, tuple): obs = obs[0]
            ep_return = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                r = float(rewards[0]) if np.ndim(rewards) else float(rewards)
                done = bool(dones[0]) if np.ndim(dones) else bool(dones)
                ep_return += r
            # simple heuristic: final ep_return sign as win/tie
            if ep_return > 0: wins += 1
            elif ep_return == 0: ties += 1
            obs = env.reset()
        win_rate = wins / max(1, self.n_episodes)
        if self.verbose:
            print(f"[eval/winrate] steps={self.n_calls} win_rate={win_rate:.3f} ties={ties}/{self.n_episodes}")
        self.logger.record("eval/win_rate_fixed", win_rate)
        return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="runs/ppo_league")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--field-size", type=float, default=1.5)
    ap.add_argument("--k-red", type=int, default=8)
    ap.add_argument("--k-green", type=int, default=4)
    ap.add_argument("--goal-reward", type=float, default=10.0)
    ap.add_argument("--red-reward", type=float, default=0.0)
    ap.add_argument("--goal-radius", type=float, default=0.10)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--fixed-bots", nargs="*", default=["SimpleBot2","ConeKeeper","TerritoryDash","AuctionStrider"])
    ap.add_argument("--selfplay-start", type=int, default=200_000)
    ap.add_argument("--selfplay-every", type=int, default=100_000)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--ckpt-freq", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--no-vecnorm", action="store_true")
    ap.add_argument("--critic-warmup", type=int, default=0)
    ap.add_argument("--input-bc-policy", type=str, default=None,
                help="Path to a pre-trained BC policy (.pth or .script.pt) to initialize PPO actor from")
    args=ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    opp_mgr = OpponentManager(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green, seed=args.seed)
    for b in args.fixed_bots:
        try: opp_mgr.add_fixed(b)
        except Exception as e: print("[warning] skipping unknown bot:", b, e)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opp_mgr.set_device(device)

    opp_mgr_fixed = OpponentManager(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green, seed=args.seed+999)
    for b in args.fixed_bots:
        try: opp_mgr_fixed.add_fixed(b)
        except Exception: pass
    opp_mgr_fixed.set_device(device)

    base_env = make_league_env(args.n_envs, opp_mgr, args)
    if not args.no_vecnorm:
        vecnorm = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        venv = vecnorm
    else:
        vecnorm = None
        venv = base_env

    policy_kwargs = dict(net_arch=dict(pi=[256,128], vf=[256,128]), activation_fn=nn.ReLU, ortho_init=True)

    model = PPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed, tensorboard_log=args.run_dir,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    if args.input_bc_policy is not None:
        print(f"[init] loading BC weights from {args.input_bc_policy}")
        state_dict = torch.load(args.input_bc_policy, map_location="cpu")
        # If it's TorchScript, you may need to jit.load instead and extract .state_dict()
        if isinstance(state_dict, dict) and "net" in state_dict:
            bc_state = state_dict["net"]
        elif isinstance(state_dict, dict):
            bc_state = state_dict
        else:
            bc_state = state_dict.state_dict()

        # Copy into PPO policy (actor only: mlp_extractor.policy_net + action_net)
        ppo_pol = model.policy
        own_sd = ppo_pol.state_dict()
        matched = {k:v for k,v in bc_state.items() if k in own_sd and own_sd[k].shape == v.shape}
        own_sd.update(matched)
        ppo_pol.load_state_dict(own_sd)
        print(f"[init] copied {len(matched)} tensors from BC into PPO actor")


    eval_env = make_league_env(1, opp_mgr_fixed, args)
    if vecnorm is not None:
        eval_vec = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_vec.load_running_stats(vecnorm); eval_env_wrapped = eval_vec
    else:
        eval_env_wrapped = eval_env

    eval_cb = EvalCallback(eval_env_wrapped, best_model_save_path=args.run_dir, eval_freq=args.eval_freq,
                           n_eval_episodes=10, deterministic=True, render=False)

    ckpt_cb = CheckpointCallback(save_freq=args.ckpt_freq // max(1,args.n_envs), save_path=args.run_dir, name_prefix="ppo_ckpt")
    snap_cb = SnapshotCallback(save_freq=max(1, args.selfplay_every // max(1,args.n_envs)),
                               save_dir=os.path.join(args.run_dir,"league"),
                               opp_mgr=opp_mgr, verbose=1)
    callbacks = [ckpt_cb, eval_cb, snap_cb]
    if vecnorm is not None:
        callbacks.append(VecNormSaveCallback(vecnorm, os.path.join(args.run_dir, "vecnorm.pkl"), freq=args.eval_freq))
    callbacks.append(WinRateCallback(opp_mgr_fixed, args, vecnorm, eval_freq=args.eval_freq, n_episodes=16))
    if args.critic_warmup > 0:
        callbacks.append(CriticWarmupCallback(args.critic_warmup))

    if args.selfplay_start > 0:
        print(f"[league] warmup {args.selfplay_start} steps before first snapshot...")
        model.learn(total_timesteps=args.selfplay_start, callback=callbacks, progress_bar=True)

    total_rest = max(0, args.timesteps - args.selfplay_start)
    if total_rest > 0:
        model.learn(total_timesteps=total_rest, callback=callbacks, progress_bar=True)

    model.save(os.path.join(args.run_dir, "ppo_final.zip"))
    if vecnorm is not None:
        vecnorm.save(os.path.join(args.run_dir, "vecnorm.pkl"))
    print("Done. Run dir:", args.run_dir)

if __name__ == "__main__":
    main()
