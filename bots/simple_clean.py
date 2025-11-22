
"""
Clean, commented versions of SimpleBot and PotentialWinner.

Drop-in compatible with the existing bot registry:
  - keep class names and __init__(index) signatures
  - keep get_controls(bot_coords, green_coords, red_coords) contract

The following utility functions and types are expected from the project:
  - From bots.utility_functions / utils.math_utils:
    get_base_coords, get_opponent_index, other_bots,
    vec_dist, vec_distTo, vec_add, vec_sub, vec_mul, vec_average,
    vec_normalize, vec_unitInDir, vec_90deg, vec_dot, vec_move,
    smoothstep, pairs, distance_to_line_segment, point_in_arena,
    vec_infnorm, steer_to_target, steer_to_target2
  - From utils.velocity_estimate: Predictor
  - From bots: register_bot

Inputs to get_controls:
  bot_coords : List[Tuple[Tuple[float, float], float]]
      For each robot (index-aligned), ((x, y), heading_radians).
  green_coords : List[Tuple[float, float]]
      Positions of "home" balls for this team (to be defended).
  red_coords : List[Tuple[float, float]]
      Positions of "away" balls (to be attacked/cleared).
Outputs from get_controls:
  Tuple[float, float]  -- left and right wheel commands (typ. in [-1, 1]).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Sequence

from bots.utility_functions import *   # noqa: F401,F403  (see module docstring)
from utils.math_utils import *         # noqa: F401,F403
from utils.velocity_estimate import Predictor
from bots import register_bot


# ---------------------------------------------------------------------------
# SimpleBot
# ---------------------------------------------------------------------------
@register_bot
class SimpleBot:
    """
    A minimal, readable baseline policy.

    Intuition
    ---------
    - Compute our distance to *own base*. Use that as a "frontier":
      * Reds (attackable) are those farther from our base than we are.
      * Greens (defendable) are those closer to our base than we are.
    - Prefer the nearest qualifying ball.
    - If no good targets exist, move to a calm "wait" position.
    - Keep a small hysteresis flag (_going_to_base) to avoid flapping.
    """
    def __init__(self, index: int) -> None:
        self._index: int = index
        self._going_to_base: bool = False

    # --- Small strategy helpers -------------------------------------------------
    def _wait_position(self) -> Tuple[float, float]:
        """A conservative staging point in our half.
        Currently symmetric for both teams; feel free to tune.
        """
        return (0.4, 1.1)

    def _own_base(self) -> Tuple[float, float]:
        return get_base_coords(self._index)

    # --- Main policy ------------------------------------------------------------
    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        own_pos = bot_coords[self._index][0]
        own_heading = bot_coords[self._index][1]

        base = self._own_base()
        dist_to_base = vec_dist(base, own_pos)

        # Hysteresis: if we're "on our way home", expand the defend zone a bit;
        # this reduces rapid switching when a ball is near the frontier.
        defend_margin = 0.5 if self._going_to_base else 0.0

        # Partition balls by whether they lie beyond or inside our base frontier.
        attackables = [p for p in red_coords if vec_dist(base, p) > dist_to_base]
        defendables = [p for p in green_coords if vec_dist(base, p) < dist_to_base - defend_margin]

        # Default target is a calm hold position (mirrored to the other diagonal
        # if the board is empty).
        target = self._wait_position()
        if len(red_coords) == 0:
            target = (target[1], target[0])

        # Choose target + update state flag
        if attackables:
            target = min(attackables, key=vec_distTo(own_pos))
            self._going_to_base = False
        elif defendables:
            target = min(defendables, key=vec_distTo(own_pos))
            self._going_to_base = True
        else:
            self._going_to_base = True

        # Convert target into differential wheel commands.
        return steer_to_target(own_pos, own_heading, target)


# ---------------------------------------------------------------------------
# PotentialWinner family
# ---------------------------------------------------------------------------
class PotentialWinnerBase:
    """
    Potential field navigator with finite-difference gradient following.

    The potential combines:
      - Mild mutual repulsion with our partner (avoid crowding).
      - Very strong repulsion near arena walls/corners.
      - Attraction to balls, scaled by how well they align with the
        vector toward the relevant goal (opponent goal for red, own
        goal for green). This prefers clearer shooting/clearing angles.
      - A modest "pair-average" repulsion from nearby *pairs* of other bots
        (discourages driving into traffic between opponents).

    Gradient estimation
      - Sample the potential slightly ahead of the robot (to focus on where
        we can actually go next), and take forward/side finite differences.
      - Convert the gradient into L/R track velocities, normalize safely.

    Notes
      - This base class can operate on the raw ball positions (param=0) or
        on predicted ball positions by a simple kinematic predictor (param=1).
    """
    # Tunable constants (feel free to tweak)
    _PARTNER_REPEL_W: float = 0.2
    _PAIR_REPEL_W: float = 2.0
    _RED_ATTR_W: float = 1.0
    _GREEN_ATTR_W: float = 0.2
    _WALL_POWER: float = 4.0
    _WALL_SCALE: float = 0.1
    _SAMPLE_AHEAD: float = 0.05      # meters ahead of the chassis to probe
    _FD_STEP: float = 0.005          # finite-difference step (m)
    _EPS: float = 1e-6               # safe normalization floor

    def __init__(self, index: int, param: int) -> None:
        self._index = index
        self._param = param          # 0 = raw positions, 1 = predicted
        self._predict = Predictor(index)

    # --- Potential function -----------------------------------------------------
    def potential_at(self, p: Tuple[float, float]) -> float:
        """Scalar potential at point p in the field.

        Lower is better. The controller drives DOWN the gradient.
        """
        opponent_base = get_base_coords(get_opponent_index(self._index))
        own_base = get_base_coords(self._index)

        pot = 0.0

        # (1) Repel from our partner to avoid crowding
        pot -= self._PARTNER_REPEL_W / vec_dist(p, self._partner_pos)

        # (2) Strong corner/wall repulsion as p approaches the borders
        #     (x, y) in [0, 1.5], steep wall growth near 0 or 1.5.
        s = self._WALL_SCALE
        w = self._WALL_POWER
        pot -= 1.0 / math.pow(max(p[0], s) / s, w)
        pot -= 1.0 / math.pow(max(p[1], s) / s, w)
        pot -= 1.0 / math.pow(max(1.5 - p[0], s) / s, w)
        pot -= 1.0 / math.pow(max(1.5 - p[1], s) / s, w)

        # (3) Repel from *pairs* of other bots based on their average position
        #     and current spacing. If two bots are nearly collocated, weight~1;
        #     as they separate past ~0.3m, the weight fades to ~0.
        for a, b in pairs(self._other_bots):
            spacing = vec_dist(a[0], b[0])
            weight = 1.0 - smoothstep(spacing / 0.3)
            center = vec_average(a[0], b[0])
            pot -= self._PAIR_REPEL_W * weight / vec_dist(p, center)

        # (4) Attraction to balls, scaled by alignment to their respective goals
        for r in self._red_positions:
            align = vec_dot(vec_normalize(vec_sub(p, opponent_base)),
                            vec_normalize(vec_sub(p, r))) + 1.0  # in [0, 2]
            pot += self._RED_ATTR_W * align / vec_dist(p, r)

        for g in self._green_positions:
            align = vec_dot(vec_normalize(vec_sub(p, own_base)),
                            vec_normalize(vec_sub(p, g))) + 1.0
            pot += self._GREEN_ATTR_W * align / vec_dist(p, g)

        return pot

    # --- Main policy ------------------------------------------------------------
    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        # Partner & environment state
        self._partner_pos = bot_coords[get_partner_index(self._index)][0]
        self._other_bots = other_bots(bot_coords, self._index)  # [(pos, angle), ...]

        # Choose which ball set to reason about
        if self._param == 0:
            self._red_positions = red_coords
            self._green_positions = green_coords
        else:
            # Use simple kinematic predictions of ball motion
            self._predict.observe(bot_coords, green_coords, red_coords)
            self._red_positions = self._predict.predict_red()
            self._green_positions = self._predict.predict_green()

        # Robot pose
        own_pos = bot_coords[self._index][0]
        own_heading = bot_coords[self._index][1]

        # Build a small local frame ahead of the robot and probe the
        # potential field for forward/side finite differences.
        forward = vec_unitInDir(own_heading)
        left = vec_90deg(forward)

        sample = vec_add(own_pos, vec_mul(forward, self._SAMPLE_AHEAD))

        d_long = self.potential_at(vec_move(sample, forward, +self._FD_STEP)) \
               - self.potential_at(vec_move(sample, forward, -self._FD_STEP))

        d_side = self.potential_at(vec_move(sample, left, +self._FD_STEP)) \
               - self.potential_at(vec_move(sample, left, -self._FD_STEP))

        # Convert gradient components into differential drive commands.
        # A positive side gradient suggests moving right track faster (turn right).
        left_track = -d_side + d_long
        right_track = +d_side + d_long

        # Normalize safely to keep outputs in a sensible range.
        scale = max(self._EPS, vec_infnorm((left_track, right_track)))
        return left_track / scale, right_track / scale


@register_bot
class PotentialWinner(PotentialWinnerBase):
    """Potential field controller on raw (measured) ball positions."""
    def __init__(self, index: int) -> None:
        super().__init__(index, param=0)


@register_bot
class PotentialWinner2(PotentialWinnerBase):
    """Potential field controller on *predicted* ball positions."""
    def __init__(self, index: int) -> None:
        super().__init__(index, param=1)
        
        
# ---------------------------------------------------------------------------
# SimpleBotGrace
# ---------------------------------------------------------------------------
@register_bot
class SimpleBotGrace:
    """
    A more advanced controller based on "A Smooth Control Law for Graceful Motion
    of Differential Wheeled Mobile Robots in 2D environment"
    Approaches the balls in more clever directions so they are aimed towards the goal
    """
    def __init__(self, index: int) -> None:
        self._index: int = index
        self._going_to_base: bool = False

    # --- Small strategy helpers -------------------------------------------------
    def _wait_position(self) -> Tuple[float, float]:
        """A conservative staging point in our half.
        Currently symmetric for both teams; feel free to tune.
        """
        return (0.4, 1.1)

    def _own_base(self) -> Tuple[float, float]:
        return get_base_coords(self._index)

    def _opp_base(self) -> Tuple[float, float]:
        return vec_sub((1.5, 1.5), self._own_base())
    
    def steer_to_target(self, own_pos, own_heading, target, target_heading):
        dx = target[0] - own_pos[0]
        dy = target[1] - own_pos[1]
        dir_target = math.atan2(dy, dx)
        theta = normalize_angle(target_heading - dir_target)
        delta = normalize_angle(own_heading - dir_target)
        r = math.hypot(dx,dy)
        k1 = 1
        k2 = 15
        delta_ref = math.atan(-k1 * theta)
        kappa=-1/r * (k2*(delta-delta_ref) + (1 + k1/(1+k1*k1*theta*theta))*math.sin(delta))
        wheel_base_half = 0.05
        max_wheel_vel = 1.0
        omega = kappa * max_wheel_vel
        vr = max_wheel_vel + omega * wheel_base_half
        vl = max_wheel_vel - omega * wheel_base_half
        max_vel = max(vr,vl)
        vl = vl / max_vel * max_wheel_vel
        vr = vr / max_vel * max_wheel_vel
        return (vl, vr)
    # --- Main policy ------------------------------------------------------------
    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        own_pos = bot_coords[self._index][0]
        own_heading = bot_coords[self._index][1]

        base = self._own_base()
        dist_to_base = vec_dist(base, own_pos)

        # Hysteresis: if we're "on our way home", expand the defend zone a bit;
        # this reduces rapid switching when a ball is near the frontier.
        defend_margin = 1.0 if self._going_to_base else 0.0

        # Partition balls by whether they lie beyond or inside our base frontier.
        attackables = [p for p in red_coords if vec_dist(base, p) > dist_to_base]
        defendables = [p for p in green_coords if vec_dist(base, p) < dist_to_base - defend_margin]

        # Default target is a calm hold position (mirrored to the other diagonal
        # if the board is empty).
        target = self._wait_position()
        if len(red_coords) == 0:
            target = (target[1], target[0])

        # Choose target + update state flag
        if red_coords:
            target = min(red_coords, key=vec_distTo(own_pos))
            self._going_to_base = False
        elif green_coords:
            target = min(green_coords, key=vec_distTo(own_pos))
            self._going_to_base = True
        else:
            self._going_to_base = True

        target_base = self._own_base() if self._going_to_base else self._opp_base() 
        target_heading = vec_angle(vec_sub(target_base, target))
        # Convert target into differential wheel commands.
        return self.steer_to_target(own_pos, own_heading, target, target_heading)
