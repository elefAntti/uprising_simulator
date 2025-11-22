
"""
AegisPilotGS — AegisPilot with Game-State & Congestion weighting
=================================================================
- Smooth, parameterized adjustments to ball attractiveness based on:
  * Score situation (reds in opp corner vs in own corner)
  * Time left in match
  * Early-game red rush
  * Corner/cluster congestion (multiple bots + low ball speed)
  * Own-goal green risk near our corner
- Uses utils.velocity_estimate.Predictor (backed by BallTracker) for stable IDs & velocities.
- Leaves navigation to the underlying potential fields; only the *weights* of attractions are modulated.

Constructor kwargs extend AegisPilot defaults with the new params (see DEFAULTS below).
"""
from __future__ import annotations
import math
from typing import List, Tuple

from bots import register_bot
from bots.utility_functions import *   # steering helpers, bases, etc.
from utils.math_utils import *

try:
    from utils.velocity_estimate import Predictor
    _HAS_PRED = True
except Exception:
    _HAS_PRED = False

def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

class _GameState:
    def __init__(self, field_size: float, goal_radius: float, time_total: float):
        self.field = field_size
        self.goal_r = goal_radius
        self.time_total = time_total
        self.t = 0.0
        self.red_in_opp = 0
        self.red_in_own = 0

    def update(self, reds, base_own, base_opp, dt):
        self.t += dt
        self.red_in_opp = sum(1 for r in reds if vec_dist(r, base_opp) <= self.goal_r)
        self.red_in_own = sum(1 for r in reds if vec_dist(r, base_own) <= self.goal_r)

    @property
    def time_left(self):
        return max(0.0, self.time_total - self.t)

    @property
    def score_diff(self):
        return self.red_in_opp - self.red_in_own  # positive if we lead

@register_bot(name="AegisPilotGS")
class AegisPilotGS:
    DEFAULTS = dict(
        # --- Base Aegis-like potentials (kept minimal for brevity) ---
        red_attr_w=1.0, green_attr_w=0.6,
        pair_repel_w=1.0, partner_repel_w=0.6,
        wall_power=4.5, wall_scale=0.12,
        sample_ahead=0.06, fd_step=0.008,
        field_size=1.5,
        use_prediction=True,
        # --- Game-state & urgency ---
        time_total=30.0,
        goal_radius=0.20,
        state_tau=10.0,
        early_red_bonus=0.35,
        early_tau=8.0,
        red_trail_gain=0.6,
        green_lead_gain=0.5,
        green_risk_scale=0.25,
        # --- Congestion ---
        cong_radius=0.18,
        cong_bot_ref=2.0,
        cong_speed_thresh=0.05,
        cong_near_w=1.0,
        cong_slow_w=1.0,
        cong_penalty_w=0.8,
        cong_penalty_max=0.7,
    )

    def __init__(self, index: int, **params):
        self.i = index
        self.cfg = dict(self.DEFAULTS); self.cfg.update({k: v for k, v in params.items() if k in self.DEFAULTS})
        self._t = 0.0
        self._state = _GameState(self.cfg['field_size'], self.cfg['goal_radius'], self.cfg['time_total'])
        self._predict = Predictor(index) if (_HAS_PRED and self.cfg['use_prediction']) else None
        # work buffers
        self._red_positions = []
        self._green_positions = []
        self._red_weights = []

    # --- Potential field pieces (minimal, illustrative) ---
    def _potentials(self, pos, ang, base_own, base_opp):
        pot_x, pot_y = 0.0, 0.0

        # Walls (soft repel)
        s = self.cfg['field_size']; sc = self.cfg['wall_scale']; pw = self.cfg['wall_power']
        for wx in (0.0, s):
            dx = pos[0] - wx
            pot_x += (1 if dx > 0 else -1) * (1.0 / max(1e-3, abs(dx))) ** pw
        for wy in (0.0, s):
            dy = pos[1] - wy
            pot_y += (1 if dy > 0 else -1) * (1.0 / max(1e-3, abs(dy))) ** pw

        # Red attractions (weighted per-ball, include alignment later during steering)
        for (r, w_red) in zip(self._red_positions, self._red_weights):
            to = vec_sub(r, pos); d = max(1e-3, vec_norm(to))
            g = vec_mul(vec_normalize(to), self.cfg['red_attr_w'] * w_red / d)
            pot_x += g[0]; pot_y += g[1]

        # Green attractions (risk-weighted near own corner to clear)
        for gpos in self._green_positions:
            to = vec_sub(gpos, pos); d = max(1e-3, vec_norm(to))
            risk = math.exp(-vec_dist(gpos, base_own) / max(1e-3, self.cfg['green_risk_scale']))
            w = self.cfg['green_attr_w'] * (1.0 + self._green_state_factor * risk)
            g = vec_mul(vec_normalize(to), w / d)
            pot_x += g[0]; pot_y += g[1]

        # Partner repel (simple)
        partner = get_partner_index(self.i)
        ppos = None
        try:
            ppos = self._all_bots[partner][0]
        except Exception:
            pass
        if ppos is not None:
            d = max(1e-3, vec_dist(pos, ppos))
            away = vec_normalize(vec_sub(pos, ppos))
            pot_x += away[0] * self.cfg['partner_repel_w'] / d
            pot_y += away[1] * self.cfg['partner_repel_w'] / d

        return (pot_x, pot_y)

    def get_controls(self, bot_coords, green_coords, red_coords):
        # Save for partner repel
        self._all_bots = bot_coords

        # Time and bases
        dt = 0.05
        self._t += dt
        base_own = get_base_coords(self.i)
        base_opp = get_base_coords(get_opponent_index(self.i))

        # Predictor for stable IDs + velocities
        reds = list(red_coords); greens = list(green_coords)
        tracks = []
        if self._predict is not None:
            try:
                self._predict.observe(bot_coords, green_coords, red_coords)
                reds = self._predict.predict_red(0.0)
                greens = self._predict.predict_green(0.0)
                tracks = self._predict.tracks_red()
            except Exception:
                pass

        # Update game state
        self._state.update(reds, base_own, base_opp, dt)

        # State factors
        U = math.exp(-self._state.time_left / max(1e-3, self.cfg['state_tau']))
        lead = max(0, self._state.score_diff)
        trail = max(0, -self._state.score_diff)
        self._red_state_factor = 1.0 + self.cfg['red_trail_gain'] * U * trail
        self._green_state_factor = 1.0 + self.cfg['green_lead_gain'] * U * lead
        early = math.exp(-self._t / max(1e-3, self.cfg['early_tau']))
        self._red_state_factor *= (1.0 + self.cfg['early_red_bonus'] * early)

        # Congestion-aware red weights per ball
        bots_all = [b[0] for b in bot_coords]
        def speed_for(r):
            if not tracks: return 0.0
            j = min(range(len(tracks)), key=lambda k: vec_dist(tracks[k]['pos'], r))
            vx, vy = tracks[j]['vel']; return math.hypot(vx, vy)
        self._red_weights = []
        for r in reds:
            near = sum(1 for p in bots_all if vec_dist(p, r) <= self.cfg['cong_radius'])
            crowd = min(1.0, near / max(1e-3, self.cfg['cong_bot_ref']))
            spd = speed_for(r)
            slow = max(0.0, 1.0 - spd / max(1e-3, self.cfg['cong_speed_thresh']))
            cong = self.cfg['cong_penalty_w'] * (self.cfg['cong_near_w']*crowd + self.cfg['cong_slow_w']*slow) / (self.cfg['cong_near_w'] + self.cfg['cong_slow_w'])
            cong = min(self.cfg['cong_penalty_max'], max(0.0, cong))
            w = self._red_state_factor * (1.0 - cong)
            self._red_weights.append(w)

        # Save lists
        self._red_positions = reds
        self._green_positions = greens

        # Convert potential into a steering target a bit ahead of the bot
        pos, ang = bot_coords[self.i]
        fx, fy = self._potentials(pos, ang, base_own, base_opp)
        # Desired heading
        desired = math.atan2(fy, fx)
        # Simple P-controller to tracks
        ang_err = (desired - ang + math.pi) % (2 * math.pi) - math.pi
        w = _clamp(2.0 * ang_err, -3.0, 3.0)
        v = 1.0  # normalized forward
        k = 0.5
        left = _clamp(v - k * w, -1.0, 1.0)
        right = _clamp(v + k * w, -1.0, 1.0)
        m = max(1.0, abs(left), abs(right))
        return left / m, right / m
