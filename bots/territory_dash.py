
"""
TerritoryDash — Voronoi zoning + dash‑hook corner handling
==========================================================
Reactive bot that:
  - Claims a territory defined by the perpendicular bisector to its partner (Voronoi cell).
  - Prioritizes red balls inside its zone (and a small band just outside) to avoid conflicts.
  - Executes a **dash‑hook** approach for balls stuck in non‑goal corners: offset, arc, and peel out.
  - Early-game bias toward reds aligned to the opponent corner.
  - Short commitment window to avoid oscillation; **handoff lock** yields to partner if they’re clearly closer.
  - Own‑goal wedge protection to avoid accidental pushes toward own corner.

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
class TerritoryDash:
    DEFAULTS = dict(
        # Zoning & selection
        zone_margin=0.08,          # m bias to prefer own side of the bisector
        near_band=0.15,            # include balls within this of the bisector
        early_bonus=0.40,          # early-game alignment bonus (decays)
        early_decay=10.0,          # seconds
        partner_eta_w=0.35,        # yield if partner ETA much better
        # Corner detection & dash-hook macro
        corner_radius=0.18,        # corners counted as stuck if within this radius (non-goal corners)
        hook_offset=0.14,          # lateral offset before approach (m)
        hook_turn_rate=2.0,        # angular speed during swing (rad/s)
        hook_speed=0.75,           # forward speed during hook
        hook_offset_time=0.34,     # s to move to offset
        hook_swing_time=0.45,      # s to swing back toward goal ray
        # Motion
        run_speed=0.95,            # nominal forward speed
        w_limit=3.0,               # max yaw rate
        # Safety
        wedge_angle_deg=40.0, wedge_radius=0.45, wedge_penalty=2.0,
        # Anti-oscillation / handoff
        commit_time=0.8,           # s lock on target before reselect unless much better appears
        switch_threshold=0.14,     # required util improvement to switch during commit
        handoff_lock=0.6,          # s lock when yielding target to partner
        # Field
        field_size=1.5,
    )

    def __init__(self, index: int, **cfg):
        self.i = index
        self.cfg = dict(self.DEFAULTS); self.cfg.update({k: v for k, v in cfg.items() if k in self.DEFAULTS})
        self._t = 0.0
        self._last_target = None
        self._lock_until = 0.0
        self._yield_until = 0.0
        # Hook macro state
        self._mode = "NORMAL"
        self._mode_until = 0.0
        self._hook_dir = +1  # swing direction (+1/-1)

    # -------------------- time --------------------
    def _now(self):
        self._t += 0.05  # crude tick-based time
        return self._t

    # -------------------- helpers --------------------
    def _bisector_side(self, p, a, b):
        """
        Signed distance to the perpendicular bisector of segment ab.
        Positive if closer to a than b.
        """
        # vector from midpoint to p projected on (a-b) direction
        mid = vec_mul(vec_add(a, b), 0.5)
        d = vec_sub(a, b)
        if vec_norm(d) < 1e-9:
            return 0.0
        n = vec_normalize(d)
        return vec_dot(vec_sub(p, mid), n)

    def _eta(self, a, b, v=0.9):
        return vec_dist(a, b) / max(1e-6, v)

    def _in_corner(self, p, goal_corner, s, r):
        corners = [(0.0, 0.0), (0.0, s), (s, 0.0), (s, s)]
        for c in corners:
            if vec_dist(p, c) < r and vec_dist(goal_corner, c) > 1e-6:
                return True
        return False

    # -------------------- main --------------------
    def get_controls(self, bot_coords, green_coords, red_coords):
        tnow = self._now()
        pos, ang = bot_coords[self.i]
        partner = bot_coords[get_partner_index(self.i)][0]
        base_opp = get_base_coords(get_opponent_index(self.i))
        base_own = get_base_coords(self.i)
        s = self.cfg['field_size']

        if not red_coords:
            # idle to a spread pose
            target = (0.25, 1.25) if self.i % 2 == 0 else (1.25, 0.25)
            return steer_to_target(pos, ang, target)

        # zone classification
        side = self._bisector_side(pos, pos, partner)  # sign reference
        # select balls in my Voronoi cell (+margin) and a near band
        cand = []
        early_weight = self.cfg['early_bonus'] * math.exp(-max(0.0, self._t) / max(1e-6, self.cfg['early_decay']))
        for idx, r in enumerate(red_coords):
            d_me = vec_dist(pos, r); d_partner = vec_dist(partner, r)
            # signed "which side": positive means closer to me in the axis from me to partner
            sd = self._bisector_side(r, pos, partner)
            in_zone = (sd >= self.cfg['zone_margin'])
            near_border = (abs(sd) < self.cfg['near_band'])
            if not in_zone and not near_border:
                continue  # leave to partner
            align = max(0.0, vec_dot(vec_normalize(vec_sub(base_opp, r)), vec_normalize(vec_sub(r, pos))))
            util = -self._eta(pos, r, v=self.cfg['run_speed'])                        - self.cfg['partner_eta_w'] * math.exp(-abs(self._eta(partner, r, v=self.cfg['run_speed']) - self._eta(pos, r, v=self.cfg['run_speed'])))                        + early_weight * align
            # hysteresis toward last target
            if self._last_target == idx:
                util += self.cfg['switch_threshold'] * 0.75
            corner = self._in_corner(r, base_opp, s, self.cfg['corner_radius'])
            cand.append((idx, r, util, corner))

        if not cand:
            # nothing in our zone; pick the globally best but with yield lock
            best = None
            for idx, r in enumerate(red_coords):
                util = -self._eta(pos, r, v=self.cfg['run_speed'])
                if best is None or util > best[0]:
                    best = (util, idx, r, False)
            _, idx, target_ball, corner = best
        else:
            idx, target_ball, _, corner = max(cand, key=lambda t: t[2])

        # commitment & handoff
        if self._last_target != idx:
            if tnow < self._lock_until:
                # keep previous unless significant improvement
                prev_idx = self._last_target
                if prev_idx is not None and prev_idx < len(red_coords):
                    # if not strictly better, stick
                    prev_r = red_coords[prev_idx]
                    prev_util = -self._eta(pos, prev_r, v=self.cfg['run_speed'])
                    new_util = -self._eta(pos, target_ball, v=self.cfg['run_speed'])
                    if (new_util - prev_util) < self.cfg['switch_threshold']:
                        idx = prev_idx; target_ball = prev_r
                    else:
                        self._last_target = idx; self._lock_until = tnow + self.cfg['commit_time']
                else:
                    self._last_target = idx; self._lock_until = tnow + self.cfg['commit_time']
            else:
                self._last_target = idx; self._lock_until = tnow + self.cfg['commit_time']

        # yield if partner is clearly closer and we're not locked
        if tnow > self._yield_until and tnow > self._lock_until:
            eta_me = self._eta(pos, target_ball, v=self.cfg['run_speed'])
            eta_pt = self._eta(partner, target_ball, v=self.cfg['run_speed'])
            if eta_pt + 0.15 < eta_me:  # partner substantially earlier
                self._yield_until = tnow + self.cfg['handoff_lock']
                # move to staging waypoint (spread)
                stage = (0.3, 1.2) if self.i % 2 == 0 else (1.2, 0.3)
                return steer_to_target(pos, ang, stage)

        # goal-ray desired direction
        desired_dir = vec_normalize(vec_sub(target_ball, pos))
        if vec_dist(target_ball, pos) < 0.14:
            desired_dir = vec_normalize(vec_sub(base_opp, target_ball))

        # own-goal wedge safety near own corner
        if vec_dist(target_ball, base_own) < self.cfg['wedge_radius']:
            own_dir = vec_normalize(vec_sub(base_own, target_ball))
            if math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(desired_dir, own_dir))))) < self.cfg['wedge_angle_deg']:
                # sidestep
                desired_dir = vec_90deg(own_dir) if (self.i % 2 == 0) else vec_mul(vec_90deg(own_dir), -1.0)

        # Corner dash-hook macro
        if self._mode == "NORMAL" and corner and vec_dist(target_ball, pos) < 0.35:
            # place an offset point lateral to ball away from the corner, then swing toward goal ray
            to_goal = vec_normalize(vec_sub(base_opp, target_ball))
            lat = vec_90deg(to_goal)
            self._hook_dir = +1 if (self.i % 2 == 0) else -1
            self._hook_point = vec_add(target_ball, vec_mul(lat, self._hook_dir * self.cfg['hook_offset']))
            self._mode = "HOOK_OFFSET"; self._mode_until = tnow + self.cfg['hook_offset_time']

        if self._mode == "HOOK_OFFSET":
            if tnow >= self._mode_until:
                self._mode = "HOOK_SWING"; self._mode_until = tnow + self.cfg['hook_swing_time']
            # drive to offset point
            return steer_to_target(pos, ang, self._hook_point)

        if self._mode == "HOOK_SWING":
            if tnow >= self._mode_until:
                self._mode = "NORMAL"
            # forward with controlled yaw toward goal ray
            k = 0.5
            goal_ray = math.atan2((base_opp[1]-target_ball[1]), (base_opp[0]-target_ball[0]))
            ang_err = (goal_ray - ang + math.pi) % (2*math.pi) - math.pi
            w = _clamp(2.0 * ang_err + self._hook_dir * 0.5, -self.cfg['w_limit'], self.cfg['w_limit'])
            v = self.cfg['hook_speed']
            L = _clamp(v - k*w, -1, 1); R = _clamp(v + k*w, -1, 1)
            return L, R

        # Normal dash controller
        desired_yaw = math.atan2(desired_dir[1], desired_dir[0])
        ang_err = (desired_yaw - ang + math.pi) % (2 * math.pi) - math.pi
        w = _clamp(2.2 * ang_err, -self.cfg['w_limit'], self.cfg['w_limit'])
        v = self.cfg['run_speed']
        k = 0.5
        left = _clamp(v - k * w, -1.0, 1.0)
        right = _clamp(v + k * w, -1.0, 1.0)
        m = max(1.0, abs(left), abs(right))
        return left / m, right / m
