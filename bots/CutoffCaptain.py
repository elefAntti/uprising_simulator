
"""
CutoffCaptain — reactive intercept + corner-sweep controller
============================================================
Goals (from spec):
- Early game: aggressively intercept/push reds toward opponent goal.
- Avoid grinding in non-goal corners: detect stalled progress and run a sweep macro.
- Avoid own-goal pushes: apply a no-go wedge near our goal to steer pushes sideways/out.
- Deconflict with partner: choose targets with overlap-aware ETA costs.
- Reduce target chatter: commit briefly to chosen ball unless a clearly better one appears.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Sequence, Optional

from bots import register_bot
from bots.utility_functions import *   # noqa: F401,F403
from utils.math_utils import *         # noqa: F401,F403

ARENA_SIZE = 1.5  # assuming [0, 1.5] x [0, 1.5]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d

def unit(v: Tuple[float, float]) -> Tuple[float, float]:
    return vec_normalize(v) if (v[0]*v[0] + v[1]*v[1]) > 1e-12 else (1.0, 0.0)

def eta_to_point(pos: Tuple[float, float], heading: float, target: Tuple[float, float], speed: float = 1.0) -> float:
    dist = vec_dist(pos, target)
    dir_vec = unit(vec_sub(target, pos))
    head_vec = vec_unitInDir(heading)
    turn = abs(math.atan2(head_vec[0]*dir_vec[1] - head_vec[1]*dir_vec[0], head_vec[0]*dir_vec[0] + head_vec[1]*dir_vec[1]))
    return dist / max(1e-6, speed) + 0.3 * turn

def corner_centers(own_base: Tuple[float, float], opp_base: Tuple[float, float]) -> List[Tuple[float, float]]:
    all_c = [(0.0, 0.0), (0.0, ARENA_SIZE), (ARENA_SIZE, 0.0), (ARENA_SIZE, ARENA_SIZE)]
    return [c for c in all_c if vec_dist(c, own_base) > 1e-6 and vec_dist(c, opp_base) > 1e-6]

def is_near(p: Tuple[float, float], q: Tuple[float, float], r: float) -> bool:
    return vec_dist(p, q) <= r

def project_progress(p_prev: Tuple[float, float], p_now: Tuple[float, float], dir_vec: Tuple[float, float]) -> float:
    d = vec_sub(p_now, p_prev)
    return vec_dot(d, unit(dir_vec))

@register_bot
class CutoffCaptain:
    def __init__(self, index: int, **params) -> None:
        self.i = index
        self.cfg = dict(
            early_bonus_frames=150,
            early_bonus=0.4,
            partner_overlap_w=0.7,
            partner_dist_w=0.2,
            commit_frames=40,
            switch_margin=0.6,
            contact_offset=0.08,
            push_ahead=0.20,
            wedge_radius=0.35,
            wedge_cos_tol=0.2,
            corner_radius=0.18,
            stall_progress_eps=0.01,
            stall_frames=28,
            sweep_back_frames=12,
            sweep_arc_frames=18,
            unstick_frames=16,
            partner_avoid_radius=0.22,
        )
        self.cfg.update({k: v for k, v in params.items() if k in self.cfg})
        self._frame = 0
        self._mode = "APPROACH"
        self._commit_left = 0
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_prev_ball: Optional[Tuple[float, float]] = None
        self._progress_acc = 0.0
        self._stall_counter = 0
        self._macro_counter = 0

    def _desired_dir(self, ball: Tuple[float, float], own_base: Tuple[float, float], opp_base: Tuple[float, float]) -> Tuple[float, float]:
        dir_vec = unit(vec_sub(opp_base, ball))
        inward = unit(vec_sub(own_base, ball))
        if vec_dist(ball, own_base) < self.cfg['wedge_radius'] and vec_dot(dir_vec, inward) > self.cfg['wedge_cos_tol']:
            tangential = (dir_vec[0] - inward[0]*vec_dot(dir_vec, inward),
                          dir_vec[1] - inward[1]*vec_dot(dir_vec, inward))
            dir_vec = unit(tangential) if tangential != (0.0, 0.0) else unit((-inward[1], inward[0]))
        return dir_vec

    def _corner_escape_dir(self, ball: Tuple[float, float], own_base: Tuple[float, float], opp_base: Tuple[float, float]) -> Tuple[float, float]:
        ng_corners = corner_centers(own_base, opp_base)
        nearest = min(ng_corners, key=lambda c: vec_dist(ball, c))
        if vec_dist(ball, nearest) < self.cfg['corner_radius']:
            interior = unit(vec_sub((ARENA_SIZE/2, ARENA_SIZE/2), nearest))
            to_opp = unit(vec_sub(opp_base, ball))
            return unit(vec_add(vec_mul(interior, 0.7), vec_mul(to_opp, 0.3)))
        return self._desired_dir(ball, own_base, opp_base)

    def _contact_point(self, ball: Tuple[float, float], dir_vec: Tuple[float, float]) -> Tuple[float, float]:
        return vec_add(ball, vec_mul(dir_vec, -self.cfg['contact_offset']))

    def _push_point(self, ball: Tuple[float, float], dir_vec: Tuple[float, float]) -> Tuple[float, float]:
        return vec_add(ball, vec_mul(dir_vec, self.cfg['push_ahead']))

    def _select_ball(self, bot_coords, red_coords) -> Optional[Tuple[float, float]]:
        if not red_coords:
            return None
        pos, heading = bot_coords[self.i]
        partner = bot_coords[get_partner_index(self.i)][0]
        own_base = get_base_coords(self.i)
        opp_base = get_base_coords(get_opponent_index(self.i))
        best_ball = None
        best_score = 1e9
        for b in red_coords:
            dir_vec = self._corner_escape_dir(b, own_base, opp_base)
            cp = self._contact_point(b, dir_vec)
            eta_self = eta_to_point(pos, heading, cp)
            eta_partner = vec_dist(partner, cp)
            overlap = math.exp(-abs(eta_self - eta_partner))
            partner_closeness = 1.0 / (1e-3 + vec_dist(pos, partner))
            penalty = self.cfg['partner_overlap_w'] * overlap + self.cfg['partner_dist_w'] * partner_closeness
            early_bonus = self.cfg['early_bonus'] if self._frame < self.cfg['early_bonus_frames'] else 0.0
            score = eta_self + penalty - early_bonus
            if score < best_score - 1e-6:
                best_score = score
                best_ball = b
        return best_ball

    def _update_progress(self, current_ball: Tuple[float, float], desired_dir: Tuple[float, float]):
        if self._target_prev_ball is None:
            self._target_prev_ball = current_ball
            self._progress_acc = 0.0
            self._stall_counter = 0
            return
        prog = project_progress(self._target_prev_ball, current_ball, desired_dir)
        self._target_prev_ball = current_ball
        if prog >= self.cfg['stall_progress_eps']:
            self._progress_acc += prog
            self._stall_counter = 0
        else:
            self._stall_counter += 1

    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:

        self._frame += 1
        pos, heading = bot_coords[self.i]
        own_base = get_base_coords(self.i)
        opp_base = get_base_coords(get_opponent_index(self.i))
        partner_pos = bot_coords[get_partner_index(self.i)][0]

        current_target = None
        if self._target_pos is not None and red_coords:
            nearest_curr = min(red_coords, key=lambda p: vec_dist(p, self._target_pos))
            if vec_dist(nearest_curr, self._target_pos) < 0.15:
                current_target = nearest_curr

        best_ball = self._select_ball(bot_coords, red_coords)
        if current_target is None:
            self._target_pos = best_ball
            self._commit_left = self.cfg['commit_frames']
            self._mode = "APPROACH"
            self._target_prev_ball = None
        else:
            if best_ball is not None:
                dir_curr = self._corner_escape_dir(current_target, own_base, opp_base)
                dir_best = self._corner_escape_dir(best_ball, own_base, opp_base)
                eta_curr = eta_to_point(pos, heading, self._contact_point(current_target, dir_curr))
                eta_best = eta_to_point(pos, heading, self._contact_point(best_ball, dir_best))
                if self._commit_left <= 0 or (eta_curr - eta_best) > self.cfg['switch_margin']:
                    self._target_pos = best_ball
                    self._commit_left = self.cfg['commit_frames']
                    self._mode = "APPROACH"
                    self._target_prev_ball = None
                else:
                    self._commit_left -= 1

        if self._target_pos is None:
            wait = (ARENA_SIZE*0.35, ARENA_SIZE*0.65) if self.i % 2 == 0 else (ARENA_SIZE*0.65, ARENA_SIZE*0.35)
            return steer_to_target(pos, heading, wait)

        des_dir = self._corner_escape_dir(self._target_pos, own_base, opp_base)

        if self._mode == "PUSH":
            self._update_progress(self._target_pos, des_dir)
            if self._stall_counter >= self.cfg['stall_frames']:
                self._mode = "SWEEP_BACK"
                self._macro_counter = self.cfg['sweep_back_frames']

        if self._mode == "SWEEP_BACK":
            left = -0.6; right = -0.3
            self._macro_counter -= 1
            if self._macro_counter <= 0:
                self._mode = "SWEEP_ARC"
                self._macro_counter = self.cfg['sweep_arc_frames']
            return (left, right)

        if self._mode == "SWEEP_ARC":
            left = 0.6; right = 0.2
            self._macro_counter -= 1
            if self._macro_counter <= 0:
                self._mode = "APPROACH"
                self._target_prev_ball = None
            return (left, right)

        partner_avoid = vec_dist(pos, partner_pos) < self.cfg['partner_avoid_radius']
        if partner_avoid:
            perp = (-des_dir[1], des_dir[0])
            sidestep = vec_add(self._target_pos, vec_mul(perp, 0.12 if self.i % 2 == 0 else -0.12))
            return steer_to_target(pos, heading, sidestep)

        dist_to_ball = vec_dist(pos, self._target_pos)
        capture_dist = self.cfg['contact_offset'] + 0.05

        if dist_to_ball > capture_dist or self._mode == "APPROACH":
            cp = self._contact_point(self._target_pos, des_dir)
            self._mode = "APPROACH"
            return steer_to_target(pos, heading, cp)

        self._mode = "PUSH"
        push_pt = self._push_point(self._target_pos, des_dir)
        return steer_to_target(pos, heading, push_pt)
