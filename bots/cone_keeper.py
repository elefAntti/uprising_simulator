
"""
ConeKeeper — Velocity-Obstacles (VO) navigation + goal-ray pushing
==================================================================
Reactive bot that:
  - Chooses a target red ball (early-game bias toward balls aligned to opponent corner).
  - Uses a VO-style sampler to pick a (v, ω) that avoids collision cones from walls/robots.
  - When close to the ball, aligns heading to the **goal ray** (ball → opponent corner) to push correctly.
  - Own-goal safety: wedge penalty near our corner to avoid pushing reds into our own goal.
  - Unstick primitive when VO is over-constrained (back-and-yaw for a few ticks).

Contract: get_controls(bot_coords, green_coords, red_coords) -> (left, right)
"""
from __future__ import annotations
import math
from typing import Sequence, Tuple, List

from bots import register_bot
from bots.utility_functions import *   # noqa: F401,F403
from utils.math_utils import *         # noqa: F401,F403

def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

@register_bot
class ConeKeeper:
    DEFAULTS = dict(
        # Sampling / kinematics
        v_max=1.0, w_max=3.0,       # max forward & angular rates (track units)
        samples_heading=9,          # headings to sample around desired
        speeds=(0.6, 1.0),          # two speed layers (low/high) in [0,1]*v_max
        horizon=0.5, dt=0.1,        # rollout horizon (s) for scoring
        # Velocity Obstacles
        safety_radius=0.12,         # clearance radius around robots (m)
        vo_time_horizon=0.7,        # lookahead for VO cones (s)
        vo_weight=1.5,              # penalty weight for VO violations
        wall_weight=1.0,            # penalty for wall proximity
        # Targeting
        engage_radius=0.12,         # distance to consider "in contact" with ball
        early_bonus=0.4, early_decay=10.0,  # early-game bias toward good alignment
        commit_time=0.7, switch_threshold=0.12,
        # Own-goal safety
        wedge_angle_deg=40.0, wedge_radius=0.45, wedge_penalty=2.0,
        # Unstick
        unstick_ticks=6, unstick_speed=0.6, unstick_turn=1.8, unstick_cooldown=0.8,
        # Field
        field_size=1.5,
    )

    def __init__(self, index: int, **cfg):
        self.i = index
        self.cfg = dict(self.DEFAULTS)
        # Allow tuple config for speeds to be overwritten
        for k, v in cfg.items():
            if k in self.cfg:
                self.cfg[k] = v
        self._t = 0.0
        self._last_target = None
        self._lock_until = 0.0
        self._unstick_timer = 0
        self._last_bad = 0

    # -------------------- Utility helpers --------------------
    def _now(self):
        self._t += 0.05  # crude tick time
        return self._t

    def _choose_target(self, pos, partner_pos, base_opp, red_coords):
        """Pick index of target red with early-game alignment bias and partner-aware cost."""
        if not red_coords:
            return None, None
        early_weight = self.cfg['early_bonus'] * math.exp(-max(0.0, self._t) / max(1e-6, self.cfg['early_decay']))
        best = None
        for idx, r in enumerate(red_coords):
            # distance/ETA
            d_me = vec_dist(pos, r)
            d_partner = vec_dist(partner_pos, r)
            eta_me = d_me / (0.8 * self.cfg['v_max'] + 1e-6)
            # alignment to opp goal (bigger is better)
            align = max(0.0, vec_dot(vec_normalize(vec_sub(base_opp, r)), vec_normalize(vec_sub(r, pos))))
            # simple corner penalty (non-goal corners)
            s = self.cfg['field_size']
            corners = [(0.0, 0.0), (0.0, s), (s, 0.0), (s, s)]
            corner_pen = 0.0
            for c in corners:
                if vec_dist(r, c) < 0.18 and vec_dist(base_opp, c) > 1e-6:
                    corner_pen = 1.0
                    break
            # partner overlap
            overlap = 1.0 / (d_partner + 1e-3)
            score = -eta_me - 0.35 * overlap - 0.5 * corner_pen + early_weight * align
            # hysteresis
            if self._last_target == idx:
                score += self.cfg['switch_threshold'] * 0.75
            if best is None or score > best[0]:
                best = (score, idx, r)
        idx = best[1]
        target_ball = best[2]
        # commitment logic
        if self._last_target != idx:
            if self._t < self._lock_until and self._last_target is not None:
                # keep previous unless significantly better
                old_idx = self._last_target
                old_util = None
                # recompute old score quickly
                r_old = red_coords[old_idx] if old_idx < len(red_coords) else target_ball
                d_me = vec_dist(pos, r_old)
                d_partner = vec_dist(partner_pos, r_old)
                eta_me = d_me / (0.8 * self.cfg['v_max'] + 1e-6)
                align = max(0.0, vec_dot(vec_normalize(vec_sub(base_opp, r_old)), vec_normalize(vec_sub(r_old, pos))))
                overlap = 1.0 / (d_partner + 1e-3)
                old_util = -eta_me - 0.35 * overlap + early_weight * align
                if best[0] - old_util < self.cfg['switch_threshold']:
                    return old_idx, r_old
            # accept switch
            self._last_target = idx
            self._lock_until = self._t + self.cfg['commit_time']
        return idx, target_ball

    def _vo_penalty(self, p, v_vec, others, dt, steps):
        """Approx VO penalty: if moving toward obstacle within time horizon and inside the cone, penalize."""
        R = self.cfg['safety_radius']
        tau = self.cfg['vo_time_horizon']
        pen = 0.0
        for q in others:
            r = vec_sub(q, p)
            dist = vec_norm(r)
            if dist < 1e-6:
                pen += 10.0; continue
            # half-angle of collision cone
            half = math.asin(min(1.0, R / max(R, dist)))
            # angle between relative position and velocity
            dir_r = vec_normalize(r)
            if vec_dot(v_vec, dir_r) <= 0:
                continue  # moving away
            ang = math.acos(max(-1.0, min(1.0, vec_dot(vec_normalize(v_vec), dir_r))))
            if ang < half:
                # estimate time-to-collision along v_vec
                closing = vec_dot(v_vec, dir_r)  # speed toward obstacle
                ttc = dist / max(1e-6, closing)
                if ttc <= tau:
                    pen += (tau - ttc) / tau + (half - ang)  # tighter & sooner → bigger penalty
        return pen * self.cfg['vo_weight']

    def _wall_penalty(self, p):
        s = self.cfg['field_size']
        dists = [p[0], p[1], s - p[0], s - p[1]]
        # penalize near walls and outside field
        pen = 0.0
        for d in dists:
            if d < 0:
                pen += 4.0 * (-d + 0.05)
            else:
                pen += max(0.0, 0.08 - d) * 6.0
        return pen * self.cfg['wall_weight']

    # -------------------- Main policy --------------------
    def get_controls(self, bot_coords, green_coords, red_coords):
        tnow = self._now()
        pos, ang = bot_coords[self.i]
        partner_pos = bot_coords[get_partner_index(self.i)][0]
        base_own = get_base_coords(self.i)
        base_opp = get_base_coords(get_opponent_index(self.i))

        # No reds: idle to spread pose
        if not red_coords:
            target = (0.25, 1.25) if self.i % 2 == 0 else (1.25, 0.25)
            return steer_to_target(pos, ang, target)

        # Select target ball with early-game bias + hysteresis
        idx, target_ball = self._choose_target(pos, partner_pos, base_opp, red_coords)

        # Desired heading: to ball, or goal-ray if very close
        to_ball = vec_sub(target_ball, pos)
        d_ball = vec_norm(to_ball)
        if d_ball < self.cfg['engage_radius']:
            # goal-ray push direction
            desired_dir = vec_normalize(vec_sub(base_opp, target_ball))
        else:
            desired_dir = vec_normalize(to_ball)

        # Own-goal wedge safety: if ball near our base, do not push into it
        if vec_dist(target_ball, base_own) < self.cfg['wedge_radius']:
            own_dir = vec_normalize(vec_sub(base_own, target_ball))
            if math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(desired_dir, own_dir))))) < self.cfg['wedge_angle_deg']:
                # rotate desired 90° away to avoid own-goal push
                desired_dir = vec_90deg(own_dir) if (self.i % 2 == 0) else vec_mul(vec_90deg(own_dir), -1.0)

        # Sample candidate (v, w) by sampling headings around desired
        samples_heading = max(3, int(self.cfg['samples_heading']))
        speeds = self.cfg['speeds'] if isinstance(self.cfg['speeds'], (list, tuple)) else (self.cfg['speeds'],)
        v_max = self.cfg['v_max']; w_max = self.cfg['w_max']
        horizon = self.cfg['horizon']; dt = self.cfg['dt']
        steps = max(1, int(horizon / dt))

        desired_yaw = math.atan2(desired_dir[1], desired_dir[0])
        # heading spread of +/- 60° around desired
        spread = math.radians(60.0)

        others = [b[0] for b in other_bots(bot_coords, self.i)]

        best = None
        for k_h in range(samples_heading):
            frac = (k_h / (samples_heading - 1)) if samples_heading > 1 else 0.5
            yaw = desired_yaw - spread + 2 * spread * frac
            # heading error to current angle → choose angular rate toward yaw
            ang_err = (yaw - ang + math.pi) % (2 * math.pi) - math.pi
            w_cmd = _clamp(2.0 * ang_err, -w_max, w_max)

            for sp in speeds:
                v_cmd = sp * v_max
                # rollout & score
                x, y, th = pos[0], pos[1], ang
                score = 0.0
                bad = 0
                for _ in range(steps):
                    # integrate unicycle
                    th += w_cmd * dt
                    vx = v_cmd * math.cos(th)
                    vy = v_cmd * math.sin(th)
                    x += vx * dt
                    y += vy * dt
                    p = (x, y)
                    vvec = (vx, vy)

                    # progress term: toward ball (if far) or along goal-ray (if close)
                    if d_ball >= self.cfg['engage_radius']:
                        score += -vec_dist(p, target_ball) * 0.6
                    else:
                        goal_ray = vec_normalize(vec_sub(base_opp, target_ball))
                        score += 0.4 * vec_dot(vec_normalize(vvec), goal_ray)

                    # VO & walls
                    vopen = self._vo_penalty(p, vvec, others, dt, steps)
                    wpen = self._wall_penalty(p)
                    score -= (vopen + wpen)
                    if vopen + wpen > 0.6:
                        bad += 1

                    # stay roughly in-field bonus
                    s = self.cfg['field_size']
                    if 0.05 < x < s - 0.05 and 0.05 < y < s - 0.05:
                        score += 0.05

                cand = (score, v_cmd, w_cmd, bad)
                if best is None or cand[0] > best[0]:
                    best = cand

        # Unstick if over-constrained consistently
        if best and best[3] >= 3:
            self._last_bad += 1
        else:
            self._last_bad = max(0, self._last_bad - 1)

        if self._last_bad >= 3 and self._unstick_timer == 0 and (self._t > self._lock_until + self.cfg['unstick_cooldown']):
            self._unstick_timer = self.cfg['unstick_ticks']

        if self._unstick_timer > 0:
            self._unstick_timer -= 1
            v_cmd = -self.cfg['unstick_speed']
            w_cmd = (1 if (self.i % 2 == 0) else -1) * self.cfg['unstick_turn']
        else:
            _, v_cmd, w_cmd, _ = best

        # Map (v,w) → differential tracks in [-1,1]
        k = 0.5
        left = _clamp(v_cmd - k * w_cmd, -1.0, 1.0)
        right = _clamp(v_cmd + k * w_cmd, -1.0, 1.0)
        m = max(1.0, abs(left), abs(right))
        return left / m, right / m
