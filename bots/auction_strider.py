
"""
AuctionStrider — reactive auction + DWA local planner
=====================================================
See docstring in previous turn for details.
"""
from __future__ import annotations
import math
from typing import Sequence, Tuple, List

from bots import register_bot
from bots.utility_functions import *   # noqa: F401,F403
from utils.math_utils import *         # noqa: F401,F403

def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

@register_bot
class AuctionStrider:
    DEFAULTS = dict(
        v_max=1.0, w_max=3.0, horizon=0.4, samples_v=4, samples_w=7,
        clearance_radius=0.12,
        early_bonus=0.4, early_decay=10.0,
        overlap_w=0.4, partner_eta_w=0.3,
        corner_penalty_w=0.6,
        commit_time=0.8, switch_threshold=0.15,
        wedge_angle_deg=40.0, wedge_radius=0.45, wedge_penalty=1.2,
        unstick_speed=0.6, unstick_turn=1.8, unstick_ticks=6, unstick_cooldown=0.8,
        field_size=1.5,
    )

    def __init__(self, index: int, **cfg):
        self.i = index
        self.cfg = dict(self.DEFAULTS); self.cfg.update({k: v for k, v in cfg.items() if k in self.DEFAULTS})
        self._last_target = None
        self._lock_until = 0.0
        self._unstick_timer = 0
        self._last_clear_bad = 0
        self._t = 0.0

    def get_controls(self, bot_coords, green_coords, red_coords):
        self._t += 0.05
        if not red_coords:
            partner = bot_coords[get_partner_index(self.i)][0]
            target = (0.25, 1.25) if self.i % 2 == 0 else (1.25, 0.25)
            if vec_dist(target, partner) < 0.25:
                away = vec_normalize(vec_sub(target, partner))
                target = vec_add(target, vec_mul(away, 0.2))
            pos, ang = bot_coords[self.i]
            return steer_to_target(pos, ang, target)

        pos, ang = bot_coords[self.i]
        partner_pos = bot_coords[get_partner_index(self.i)][0]
        base_own = get_base_coords(self.i)
        base_opp = get_base_coords(get_opponent_index(self.i))

        def eta_from(a, p):
            d = vec_dist(a, p)
            return d / (self.cfg['v_max'] * 0.8 + 1e-6)

        early_weight = self.cfg['early_bonus'] * math.exp(-max(0.0, self._t) / max(1e-6, self.cfg['early_decay']))

        candidates: List[Tuple[int, Tuple[float, float], float]] = []
        for idx, r in enumerate(red_coords):
            to_goal = vec_normalize(vec_sub(base_opp, r))
            to_me = vec_normalize(vec_sub(r, pos))
            align = max(0.0, vec_dot(to_goal, to_me))

            corner_pen = 0.0
            s = self.cfg['field_size']
            corners = [(0.0, 0.0), (0.0, s), (s, 0.0), (s, s)]
            goal_corner = base_opp
            for c in corners:
                if vec_dist(r, c) < 0.18 and vec_dist(goal_corner, c) > 1e-6:
                    corner_pen = 1.0
                    break

            eta_me = eta_from(pos, r)
            eta_partner = eta_from(partner_pos, r)
            overlap = 1.0 / (vec_dist(partner_pos, r) + 1e-3)
            util = -eta_me                        - self.cfg['overlap_w'] * overlap                        - self.cfg['partner_eta_w'] * math.exp(-abs(eta_partner - eta_me))                        - self.cfg['corner_penalty_w'] * corner_pen                        + early_weight * align

            if self._last_target == idx:
                util += self.cfg['switch_threshold'] * 0.75

            candidates.append((idx, r, util))

        idx, target_ball, _ = max(candidates, key=lambda t: t[2])

        if self._last_target != idx:
            if self._t < self._lock_until:
                old_util = [u for j, _, u in candidates if j == self._last_target]
                best_util = max(candidates, key=lambda t: t[2])[2]
                if old_util and best_util - old_util[0] < self.cfg['switch_threshold']:
                    idx = self._last_target
                    target_ball = red_coords[idx]
            else:
                self._lock_until = self._t + self.cfg['commit_time']
                self._last_target = idx

        v_max = self.cfg['v_max']; w_max = self.cfg['w_max']
        horizon = self.cfg['horizon']
        samples_v = max(1, int(self.cfg['samples_v']))
        samples_w = max(1, int(self.cfg['samples_w']))

        others = [b[0] for b in other_bots(bot_coords, self.i)]

        def clearance_penalty(p):
            pen = 0.0
            for q in others:
                d = vec_dist(p, q)
                if d < self.cfg['clearance_radius']:
                    pen += (self.cfg['clearance_radius'] - d) * 4.0
            s = self.cfg['field_size']
            wall_dists = [p[0], p[1], s - p[0], s - p[1]]
            pen += sum(max(0.0, 0.08 - d) * 6.0 for d in wall_dists)
            return pen

        best = None
        for i_v in range(samples_v):
            v = (i_v / (samples_v - 1) if samples_v > 1 else 1.0) * v_max
            for i_w in range(samples_w):
                w = -w_max + (i_w / (samples_w - 1) if samples_w > 1 else 0.5) * 2*w_max
                x, y = pos
                th = ang
                dt = 0.1
                steps = max(1, int(horizon / dt))
                score = 0.0
                bad_clear = 0
                for _ in range(steps):
                    th += w * dt
                    x += v * math.cos(th) * dt
                    y += v * math.sin(th) * dt
                    p = (x, y)
                    dist = vec_dist(p, target_ball)
                    score += -dist * 0.5
                    cpen = clearance_penalty(p)
                    score -= cpen
                    if cpen > 0.5: bad_clear += 1
                    if vec_dist(target_ball, base_own) < self.cfg['wedge_radius']:
                        goal_dir = vec_normalize(vec_sub(base_own, target_ball))
                        push_dir = vec_unitInDir(th)
                        ang_ok = math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(push_dir, goal_dir)))))
                        if ang_ok < self.cfg['wedge_angle_deg']:
                            score -= self.cfg['wedge_penalty']
                to_goal = vec_normalize(vec_sub(base_opp, target_ball))
                push_dir0 = vec_unitInDir(ang)
                score += early_weight * max(0.0, vec_dot(push_dir0, to_goal))
                if best is None or score > best[0]:
                    best = (score, v, w, bad_clear)

        if best and best[3] >= 3:
            self._last_clear_bad += 1
        else:
            self._last_clear_bad = max(0, self._last_clear_bad - 1)

        if self._last_clear_bad >= 3 and self._unstick_timer == 0 and (self._t > self._lock_until + self.cfg['unstick_cooldown']):
            self._unstick_timer = self.cfg['unstick_ticks']

        if getattr(self, "_unstick_timer", 0) > 0:
            self._unstick_timer -= 1
            v_cmd = -self.cfg['unstick_speed']; w_cmd = (1 if (self.i % 2 == 0) else -1) * self.cfg['unstick_turn']
        else:
            _, v_cmd, w_cmd, _ = best

        k = 0.5
        left = _clamp(v_cmd - k * w_cmd, -1.0, 1.0)
        right = _clamp(v_cmd + k * w_cmd, -1.0, 1.0)
        m = max(1.0, abs(left), abs(right))
        return left / m, right / m
