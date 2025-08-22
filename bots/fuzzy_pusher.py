
"""
FuzzyPusher — Mamdani fuzzy controller for (v, ω)
=================================================
Inputs: angle_to_ball, goal_alignment, corner_proximity, partner_eta_gap, progress_rate
Outputs: forward_speed v in [0,1], turn_rate ω in [-w_max,w_max]
"""
from __future__ import annotations
import math
from typing import Sequence, Tuple, List

from bots import register_bot
from bots.utility_functions import *   # noqa: F401,F403
from utils.math_utils import *         # noqa: F401,F403
try:
    from utils.velocity_estimate import Predictor
    _HAS_PRED = True
except Exception:
    _HAS_PRED = False

Vec2 = Tuple[float, float]
def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

# --- simple fuzzy shapes ---
def tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x - a)/(b - a) if x < b else (c - x)/(c - b)

def trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if a < x < b: return (x - a)/(b - a)
    return (d - x)/(d - c)

@register_bot
class FuzzyPusher:
    DEFAULTS = dict(
        w_max=3.0, v_max=1.0,
        # own-goal safety
        wedge_angle_deg=40.0, wedge_radius=0.45,
        # commitment
        commit_time=0.7, switch_threshold=0.15,
        # corner & progress
        corner_radius=0.18, field_size=1.5,
        stall_eps=0.04, stall_time=0.8,
        # predictor
        use_prediction=True, dt_ahead=0.05,
    )

    def __init__(self, index: int, **cfg):
        self.i = index
        self.cfg = dict(self.DEFAULTS); self.cfg.update({k: v for k, v in cfg.items() if k in self.DEFAULTS})
        self._t = 0.0
        self._last_idx = None
        self._last_pos = None
        self._lock_until = 0.0
        self._last_proj = None
        self._predict = None #Predictor(index, dt_ahead=self.cfg['dt_ahead']) if (_HAS_PRED and self.cfg['use_prediction']) else None

    def _now(self): self._t += 0.05; return self._t

    def _choose_target(self, pos: Vec2, partner: Vec2, base_opp: Vec2, reds: List[Vec2]):
        best = None
        for idx, r in enumerate(reds):
            eta = vec_dist(pos, r)
            align = max(0.0, vec_dot(vec_normalize(vec_sub(base_opp, r)), vec_normalize(vec_sub(r, pos))))
            sc = -eta + 0.6*align - 0.4*(1.0/(vec_dist(partner, r)+1e-3))
            if self._last_idx == idx: sc += self.cfg['switch_threshold']*0.75
            if best is None or sc > best[0]: best = (sc, idx, r)
        if best is None: return None, None
        idx, r = best[1], best[2]
        if self._last_idx != idx:
            if self._t < self._lock_until and self._last_pos is not None:
                prev = self._last_pos
                if (-vec_dist(pos, r)) - (-vec_dist(pos, prev)) < self.cfg['switch_threshold']:
                    return self._last_idx, self._last_pos
            self._last_idx, self._last_pos = idx, r
            self._lock_until = self._t + self.cfg['commit_time']
        return idx, r

    def _in_corner(self, p: Vec2, goal_corner: Vec2):
        s = self.cfg['field_size']
        for c in [(0,0), (0,s), (s,0), (s,s)]:
            if vec_dist(p, c) < self.cfg['corner_radius'] and vec_dist(goal_corner, c) > 1e-6:
                return True
        return False

    def _fuzzy(self, ang_err: float, goal_align: float, corner_near: float, partner_gap: float, progress: float):
        """
        ang_err: radians in [-pi, pi], positive = turn left.
        goal_align: dot in [-1,1] (heading vs goal ray).
        corner_near: distance in [0, 1] where 1=very near.
        partner_gap: (eta_partner - eta_self) (positive = I'm sooner).
        progress: rate along goal ray (m/s), normalized to ~[-0.3, 0.3].
        """
        # Normalize inputs
        ang = ang_err
        gal = goal_align
        cn = _clamp(corner_near, 0.0, 1.0)
        pg = _clamp(progress*2.0, -1.0, 1.0)  # scale

        # Memberships
        L = trap(ang,  math.radians(-180), math.radians(-90), math.radians(-30), math.radians(-5))
        C = tri(ang,  math.radians(-10), 0.0, math.radians(10))
        R = trap(ang,  math.radians(5), math.radians(30), math.radians(90), math.radians(180))

        Gbad = trap(gal, -1.0, -0.7, -0.3, -0.1)
        Gok  = tri(gal, -0.2, 0.0, 0.2)
        Ggood= trap(gal, 0.1, 0.3, 0.7, 1.0)

        Cnear= trap(cn, 0.6, 0.75, 1.0, 1.2)
        Cfar = trap(cn, -0.2, 0.0, 0.3, 0.5)

        Pmine= trap(pg, 0.1, 0.25, 1.0, 1.2)   # I'm making progress
        Pstall=trap(pg, -1.2, -1.0, -0.15, 0.0)

        # partner gap: >0 I am sooner, <0 partner sooner
        Hmine= trap(partner_gap, 0.05, 0.2, 1.0, 1.2)
        Htheirs= trap(partner_gap, -1.2, -1.0, -0.2, -0.05)

        # Rules -> output (v in [0,1], w in [-1,1] before scaling)
        v_acts = []
        w_acts = []

        def add(rule_strength, v, w):
            v_acts.append((rule_strength, v))
            w_acts.append((rule_strength, w))

        # Core steering: angle dominates heading
        add(R,  0.8,  +1.0)   # need to turn left -> positive w
        add(L,  0.8,  -1.0)   # need to turn right -> negative w
        add(C,  1.0,   0.0)   # on target -> go straight fast

        # Goal alignment boosts speed if not near corner
        add(min(Ggood, Cfar), 1.0, 0.0)
        add(min(Gbad, Cfar),  0.5, 0.0)

        # Corner near -> slow and add lateral turn to peel
        add(Cnear, 0.4, 0.6 if R > L else -0.6)

        # Progress stalled -> add stronger yaw, reduce v
        add(Pstall, 0.4, 0.8 if R > L else -0.8)

        # Partner sooner -> yield (lower v)
        add(Htheirs, 0.3, 0.0)

        # I'm sooner and aligned -> push harder
        add(min(Hmine, Ggood), 1.0, 0.0)

        # Defuzzify by weighted average (center of gravity)
        def defuzz(pairs, lo, hi):
            num = sum(mu * val for (mu, val) in pairs)
            den = sum(mu for (mu, _) in pairs) + 1e-9
            x = num / den
            return _clamp(x, lo, hi)

        v = defuzz(v_acts, 0.0, 1.0)
        w = defuzz(w_acts, -1.0, 1.0)
        return v, w

    def get_controls(self, bot_coords, green_coords, red_coords):
        tnow = self._now()
        pos, ang = bot_coords[self.i]
        partner = bot_coords[get_partner_index(self.i)][0]
        own_base = get_base_coords(self.i)
        opp_base = get_base_coords(get_opponent_index(self.i))

        reds = red_coords
        if self._predict is not None:
            try:
                self._predict.observe(bot_coords, green_coords, red_coords)
                reds = self._predict.predict_red()
            except Exception:
                reds = red_coords

        if not reds:
            stage = (0.25, 1.25) if self.i % 2 == 0 else (1.25, 0.25)
            return steer_to_target(pos, ang, stage)

        idx, ball = self._choose_target(pos, partner, opp_base, reds)
        # progress for stall
        goal_ray = vec_normalize(vec_sub(opp_base, ball))
        proj = vec_dot(vec_sub(ball, pos), goal_ray)
        if self._last_proj is None: self._last_proj = (tnow, proj)
        else: self._last_proj = (tnow, proj)

        # Inputs
        to_ball = vec_sub(ball, pos)
        ang_err = (math.atan2(to_ball[1], to_ball[0]) - ang + math.pi) % (2*math.pi) - math.pi
        # heading vs goal ray
        goal_yaw = math.atan2(goal_ray[1], goal_ray[0])
        head_align = vec_dot(vec_unitInDir(ang), goal_ray)

        # corner proximity normalized
        s = self.cfg['field_size']
        d_corner = min(vec_dist(ball, (0,0)), vec_dist(ball, (0,s)), vec_dist(ball, (s,0)), vec_dist(ball, (s,s)))
        # exclude goal corner
        d_corner = min([vec_dist(ball, c) for c in [(0,0),(0,s),(s,0),(s,s)] if vec_dist(opp_base, c) > 1e-6])
        c_near = _clamp((self.cfg['corner_radius'] - d_corner) / max(1e-6, self.cfg['corner_radius']), 0.0, 1.0)

        # partner gap
        gap = (vec_dist(partner, ball) - vec_dist(pos, ball)) / max(1e-6, 1.0)

        # progress rate (approx derivative of proj)
        # Note: bots don't receive dt; assume ~0.05 s tick
        prog = 0.0
        # (optional, could compute from last_proj if desired)

        v01, w01 = self._fuzzy(ang_err, head_align, c_near, gap, prog)

        # Own-goal wedge safety: if ball near our base and we point into it, zero v and turn sideways
        if vec_dist(ball, own_base) < self.cfg['wedge_radius']:
            own_dir = vec_normalize(vec_sub(own_base, ball))
            if math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(vec_unitInDir(ang), own_dir))))) < self.cfg['wedge_angle_deg']:
                v01 *= 0.3
                w01 = (1.0 if (self.i % 2 == 0) else -1.0)

        # Map to tracks
        v = v01 * self.cfg['v_max']
        w = w01 * self.cfg['w_max']
        k = 0.5
        left = _clamp(v - k*w, -1.0, 1.0)
        right = _clamp(v + k*w, -1.0, 1.0)
        m = max(1.0, abs(left), abs(right))
        return left/m, right/m
