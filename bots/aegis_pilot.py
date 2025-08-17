
"""
AegisPilot — parameterized potential-field controller
=====================================================
A marketable, drop-in replacement/variant of "PotentialWinner" that:
  - uses a transparent config dict (self.cfg) for all weights/shapes
  - accepts **params overrides in the constructor
  - optionally uses simple ball-motion prediction (use_prediction=True)
  - is registered via the decorator registry (@register_bot)

Expected project utilities:
  - bots.register_bot (decorator-based registry)
  - bots.utility_functions, utils.math_utils for vector helpers
  - utils.velocity_estimate.Predictor for simple kinematic prediction
"""
from __future__ import annotations

import math
from typing import List, Tuple, Sequence

# Project imports (wildcards for brevity; swap to explicit if you prefer)
from bots import register_bot
from bots.utility_functions import *  # noqa: F401,F403
from utils.math_utils import *        # noqa: F401,F403
try:
    from utils.velocity_estimate import Predictor
    _PREDICTOR_AVAILABLE = True
except Exception:
    _PREDICTOR_AVAILABLE = False

class _AegisBase:
    """Potential-field navigator with finite-difference steering.

    Config keys (overridable via **params):
        partner_repel_w : weight for gentle separation from partner
        pair_repel_w    : weight for avoiding the *midpoint* of any bot pair
        red_attr_w      : attraction weight to red balls (clear them away)
        green_attr_w    : attraction to green (defensive cleanups)
        wall_power      : exponent for wall/corner repulsion growth
        wall_scale      : softness scale (meters) for wall distance
        sample_ahead    : forward offset for probing potential (m)
        fd_step         : finite-difference step (m)
        use_prediction  : bool; if True and Predictor available, use predicted ball positions
    """

    # Sensible defaults (tuned for a 1.5m x 1.5m field; tweak freely)
    DEFAULTS = dict(
        partner_repel_w=0.2,
        pair_repel_w=2.0,
        red_attr_w=1.0,
        green_attr_w=0.2,
        wall_power=4.0,
        wall_scale=0.10,
        sample_ahead=0.05,
        fd_step=0.005,
        use_prediction=False,
    )

    def __init__(self, index: int, **params) -> None:
        self.index = index
        self.cfg = dict(self.DEFAULTS)
        # Allow overrides for known keys only
        for k, v in params.items():
            if k in self.cfg:
                self.cfg[k] = v
        self._predict = Predictor(index) if (self.cfg.get("use_prediction") and _PREDICTOR_AVAILABLE) else None

    # --- Potential function -------------------------------------------------
    def _potential_at(self, p: Tuple[float, float]) -> float:
        """Lower is better; we will drive DOWN the gradient."""
        opponent_base = get_base_coords(get_opponent_index(self.index))
        own_base = get_base_coords(self.index)

        pot = 0.0

        # (1) Gentle partner separation
        pot -= self.cfg['partner_repel_w'] / max(1e-6, vec_dist(p, self._partner_pos))

        # (2) Wall/corner repulsion
        s = self.cfg['wall_scale']
        w = self.cfg['wall_power']
        # arena assumed ~[0,1.5]x[0,1.5]; adjust if different
        pot -= 1.0 / math.pow(max(p[0], s) / s, w)
        pot -= 1.0 / math.pow(max(p[1], s) / s, w)
        pot -= 1.0 / math.pow(max(1.5 - p[0], s) / s, w)
        pot -= 1.0 / math.pow(max(1.5 - p[1], s) / s, w)

        # (3) Pairwise traffic avoidance via midpoint of other-bot pairs
        for a, b in pairs(self._other_bots):
            spacing = vec_dist(a[0], b[0])
            weight = 1.0 - smoothstep(spacing / 0.3)  # fade with distance
            center = vec_average(a[0], b[0])
            pot -= self.cfg['pair_repel_w'] * weight / max(1e-6, vec_dist(p, center))

        # (4) Ball attraction, modulated by goal alignment
        for r in self._red_positions:
            align = vec_dot(vec_normalize(vec_sub(p, opponent_base)),
                            vec_normalize(vec_sub(p, r))) + 1.0  # [0,2]
            pot += self.cfg['red_attr_w'] * align / max(1e-6, vec_dist(p, r))

        for g in self._green_positions:
            align = vec_dot(vec_normalize(vec_sub(p, own_base)),
                            vec_normalize(vec_sub(p, g))) + 1.0
            pot += self.cfg['green_attr_w'] * align / max(1e-6, vec_dist(p, g))

        return pot

    # --- Policy -------------------------------------------------------------
    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        # Partner & environment
        self._partner_pos = bot_coords[get_partner_index(self.index)][0]
        self._other_bots = other_bots(bot_coords, self.index)

        # Choose ball set: raw or predicted
        if self._predict is None:
            self._red_positions = red_coords
            self._green_positions = green_coords
        else:
            self._predict.observe(bot_coords, green_coords, red_coords)
            self._red_positions = self._predict.predict_red()
            self._green_positions = self._predict.predict_green()

        # Robot pose
        own_pos, own_heading = bot_coords[self.index]

        # Build a small local frame ahead of the robot and probe the potential
        forward = vec_unitInDir(own_heading)
        left = vec_90deg(forward)

        sample = vec_add(own_pos, vec_mul(forward, self.cfg['sample_ahead']))

        d_long = self._potential_at(vec_move(sample, forward, +self.cfg['fd_step'])) \
                 - self._potential_at(vec_move(sample, forward, -self.cfg['fd_step']))
        d_side = self._potential_at(vec_move(sample, left, +self.cfg['fd_step'])) \
                 - self._potential_at(vec_move(sample, left, -self.cfg['fd_step']))

        # Convert gradient into differential track speeds
        left_track = -d_side + d_long
        right_track = +d_side + d_long

        # Normalize for stability
        scale = max(1e-6, vec_infnorm((left_track, right_track)))
        return left_track / scale, right_track / scale


@register_bot
class AegisPilot(_AegisBase):
    """AegisPilot — potential-field controller with parameter overrides.

    Examples
    --------
    # Registry usage (index assigned by simulator)
    bot = AegisPilot(0)  # defaults
    tuned = AegisPilot(1, red_attr_w=1.5, wall_power=5.0, use_prediction=True)
    """
    def __init__(self, index: int, **params):
        super().__init__(index, **params)
