
"""
FieldMarshal — two-layer (mode-shaped) potential fields
=======================================================
Global mode picks a shaping (Attack / CornerRescue / OwnGoalClear / Idle),
local controller is a smooth potential field with mode-specific weights.
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

@register_bot
class FieldMarshal:
    # Base potential defaults
    DEFAULTS = dict(
        # Core field parameters
        wall_scale=0.10, wall_power=4.0,
        partner_repel_w=0.25, pair_repel_w=1.8,
        red_attr_w=1.0, green_attr_w=0.2,
        sample_ahead=0.05, fd_step=0.005,
        # Mode shaping multipliers
        ATTACK_red=1.2, ATTACK_wall=1.0, ATTACK_pair=1.0,
        CORNER_red=1.1, CORNER_wall=1.4, CORNER_pair=1.2,
        OWN_red=0.6,   OWN_wall=1.2,   OWN_pair=1.0,
        # Threat & corner logic
        wedge_angle_deg=40.0, wedge_radius=0.45,
        corner_radius=0.18, field_size=1.5,
        # Target selection / anti-chatter
        commit_time=0.7, switch_threshold=0.15,
        partner_eta_w=0.35, overlap_w=0.35, early_bonus=0.4, early_decay=10.0,
        # Stall detection for corner rescue
        stall_eps=0.04, stall_time=0.8,
        # Prediction
        use_prediction=True, dt_ahead=0.05,
    )

    def __init__(self, index: int, **cfg):
        self.i = index
        self.cfg = dict(self.DEFAULTS); self.cfg.update({k: v for k, v in cfg.items() if k in self.DEFAULTS})
        self._mode = "ATTACK"
        self._t = 0.0
        self._last_idx = None
        self._last_pos = None
        self._lock_until = 0.0
        self._last_proj = None
        self._predict = None #Predictor(index, dt_ahead=self.cfg['dt_ahead']) if (_HAS_PRED and self.cfg['use_prediction']) else None

    # --- Helpers ------------------------------------------------------------
    def _now(self): self._t += 0.05; return self._t

    def _eta(self, a: Vec2, b: Vec2, v=0.9):
        return vec_dist(a, b) / max(1e-6, v)

    def _choose_target(self, pos: Vec2, partner: Vec2, base_opp: Vec2, reds: List[Vec2]):
        early = self.cfg['early_bonus'] * math.exp(-max(0.0, self._t)/max(1e-6, self.cfg['early_decay']))
        best = None
        for idx, r in enumerate(reds):
            align = max(0.0, vec_dot(vec_normalize(vec_sub(base_opp, r)), vec_normalize(vec_sub(r, pos))))
            util = -self._eta(pos, r) - self.cfg['partner_eta_w'] * math.exp(-abs(self._eta(partner, r) - self._eta(pos, r)))                        - self.cfg['overlap_w'] * (1.0 / (vec_dist(partner, r) + 1e-3)) + early * align
            if self._last_idx == idx: util += self.cfg['switch_threshold'] * 0.75
            if best is None or util > best[0]: best = (util, idx, r)
        if best is None: return None, None
        idx, r = best[1], best[2]
        # commit logic
        if self._last_idx != idx:
            if self._t < self._lock_until and self._last_pos is not None:
                # keep previous if not significantly better
                prev = self._last_pos
                if ( -self._eta(pos, r) ) - ( -self._eta(pos, prev) ) < self.cfg['switch_threshold']:
                    return self._last_idx, self._last_pos
            self._last_idx, self._last_pos = idx, r
            self._lock_until = self._t + self.cfg['commit_time']
        return idx, r

    def _own_goal_threat(self, base_own: Vec2, reds: List[Vec2]):
        for r in reds:
            if vec_dist(r, base_own) < self.cfg['wedge_radius']:
                return r
        return None

    def _in_corner(self, p: Vec2, goal_corner: Vec2):
        s = self.cfg['field_size']
        for c in [(0,0), (0,s), (s,0), (s,s)]:
            if vec_dist(p, c) < self.cfg['corner_radius'] and vec_dist(goal_corner, c) > 1e-6:
                return True
        return False

    # Potential function with mode shaping
    def _potential(self, p: Vec2, mode: str, partner_pos: Vec2, others: List[Tuple[Vec2,float]], reds: List[Vec2], greens: List[Vec2], bases: Tuple[Vec2,Vec2], focus: Vec2|None):
        own_base, opp_base = bases
        c = self.cfg
        # multipliers per mode
        if mode == "ATTACK": mw = (c['ATTACK_red'], c['ATTACK_wall'], c['ATTACK_pair'])
        elif mode == "CORNER": mw = (c['CORNER_red'], c['CORNER_wall'], c['CORNER_pair'])
        elif mode == "OWN": mw = (c['OWN_red'], c['OWN_wall'], c['OWN_pair'])
        else: mw = (1.0,1.0,1.0)
        red_mul, wall_mul, pair_mul = mw

        pot = 0.0
        # partner separation
        pot -= c['partner_repel_w'] / max(1e-6, vec_dist(p, partner_pos))
        # wall repulsion
        s = c['field_size']; sc = c['wall_scale']; pw = c['wall_power']
        pot -= wall_mul*( 1.0/math.pow(max(p[0],sc)/sc,pw) + 1.0/math.pow(max(p[1],sc)/sc,pw)
                        + 1.0/math.pow(max(s-p[0],sc)/sc,pw) + 1.0/math.pow(max(s-p[1],sc)/sc,pw) )
        # pair avoidance (approximate using other bot midpoints)
        for a in others:
            center = a[0]
            pot -= pair_mul*c['pair_repel_w'] / max(1e-6, vec_dist(p, center))

        # ball attraction
        reds_use = reds
        if focus is not None:
            # emphasize focused target
            reds_use = [focus] + [r for r in reds if r is not focus]
        for r in reds_use:
            align = vec_dot(vec_normalize(vec_sub(opp_base, r)), vec_normalize(vec_sub(p, r))) + 1.0
            w = c['red_attr_w'] * red_mul * (1.8 if r is focus else 1.0)
            pot += w * align / max(1e-6, vec_dist(p, r))

        for g in greens:
            align = vec_dot(vec_normalize(vec_sub(own_base, g)), vec_normalize(vec_sub(p, g))) + 1.0
            pot += c['green_attr_w'] * align / max(1e-6, vec_dist(p, g))

        return pot

    # --- Policy -------------------------------------------------------------
    def get_controls(self, bot_coords, green_coords, red_coords):
        tnow = self._now()
        own_pos, own_ang = bot_coords[self.i]
        partner_pos = bot_coords[get_partner_index(self.i)][0]
        own_base = get_base_coords(self.i)
        opp_base = get_base_coords(get_opponent_index(self.i))

        reds = red_coords; greens = green_coords
        if self._predict is not None:
            try:
                self._predict.observe(bot_coords, green_coords, red_coords)
                reds = self._predict.predict_red()
                greens = self._predict.predict_green()
            except Exception:
                reds = red_coords; greens = green_coords

        if not reds:
            # idle/stage
            stage = (0.25, 1.25) if self.i % 2 == 0 else (1.25, 0.25)
            return steer_to_target(own_pos, own_ang, stage)

        # Mode selection
        threat = self._own_goal_threat(own_base, reds)
        if threat is not None:
            self._mode = "OWN"
            focus_idx, focus_ball = None, threat
        else:
            idx, ball = self._choose_target(own_pos, partner_pos, opp_base, reds)
            focus_idx, focus_ball = idx, ball
            # stall detector -> corner mode
            goal_ray = vec_normalize(vec_sub(opp_base, focus_ball))
            proj = vec_dot(vec_sub(focus_ball, own_pos), goal_ray)
            if self._last_proj is None: self._last_proj = (tnow, proj)
            else:
                tp, pp = self._last_proj
                if (tnow - tp) >= self.cfg['stall_time'] and (proj - pp) < self.cfg['stall_eps'] and self._in_corner(focus_ball, opp_base):
                    self._mode = "CORNER"
                else:
                    self._mode = "ATTACK"
                self._last_proj = (tnow, proj)

        # Potential sampling / gradient
        forward = vec_unitInDir(own_ang)
        left = vec_90deg(forward)
        sample = vec_add(own_pos, vec_mul(forward, self.cfg['sample_ahead']))

        others_simple = [b for b in other_bots(bot_coords, self.i)]  # [(pos,ang),...]
        others_simple = [(b[0], b[1]) for b in others_simple]

        def pot(pt):
            return self._potential(pt, "OWN" if self._mode=="OWN" else ("CORNER" if self._mode=="CORNER" else "ATTACK"),
                                   partner_pos, others_simple, reds, greens, (own_base, opp_base), focus_ball)

        fd = self.cfg['fd_step']
        d_long = pot(vec_move(sample, forward, +fd)) - pot(vec_move(sample, forward, -fd))
        d_side = pot(vec_move(sample, left, +fd)) - pot(vec_move(sample, left, -fd))

        # Convert gradient to tracks (down the gradient)
        left_track = -d_side + d_long
        right_track = +d_side + d_long

        # Mode tweaks
        if self._mode == "CORNER":
            # add a lateral bias to peel off the wall
            bias = 0.25 if (self.i % 2 == 0) else -0.25
            left_track += bias; right_track -= bias
        elif self._mode == "OWN":
            # clamp forward if pushing into own corner (safety)
            own_dir = vec_normalize(vec_sub(own_base, focus_ball))
            if math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(forward, own_dir))))) < self.cfg['wedge_angle_deg']:
                left_track *= 0.5; right_track *= 0.5

        # Normalize
        s = max(1e-6, max(abs(left_track), abs(right_track)))
        return left_track/s, right_track/s
