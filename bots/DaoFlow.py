from __future__ import annotations

import math
from typing import List, Tuple, Sequence

# Project imports (wildcards for brevity; swap to explicit if you prefer)
from bots import register_bot
from bots.utility_functions import *  # noqa: F401,F403
from utils.math_utils import *        # noqa: F401,F403

@register_bot
class DaoFlow:
    DEFAULTS=dict(
            attackable_base_score = 10.0,
            defendable_base_score = 5.0,
            attackable_distance_weight = -2.0,
            defendable_distance_weight = -2.0,
            attackable_friend_distance_weight = 1.0,
            defendable_friend_distance_weight = 1.0,
            attackable_goal_distance_weight = -0.5,
            defendale_goal_distance_weight = -0.5,
            corner_safety_weight = 2.0,
            corner_safety_scale = 3.0,
            hysteresis = 0.5
    )
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
    def __init__(self, index: int, **params) -> None:
        self.cfg = dict(self.DEFAULTS)
        # Allow overrides for known keys only
        for k, v in params.items():
            if k in self.cfg:
                self.cfg[k] = v
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

    def _min_dist_to_neutral_corner(self, pos):
        return min(vec_dist(pos, (0.0, 1.5)), vec_dist(pos, (1.5, 0.0)))

    def _get_friend_index(self):
        return [1,0,3,2][self._index]

    def score_attackable(self, own_pos, friend_pos, goal_pos):
        return lambda pos: (self.cfg["attackable_base_score"] \
                + self.cfg["attackable_distance_weight"] * vec_dist(pos, own_pos) \
                + self.cfg["attackable_friend_distance_weight"] * vec_dist(pos, friend_pos) \
                + self.cfg["attackable_friend_distance_weight"] * vec_dist(pos, goal_pos)  \
                + smoothstep(self._min_dist_to_neutral_corner(pos) * self.cfg["corner_safety_scale"]) * self.cfg["corner_safety_weight"])\
                if pos else -10000

    def score_defendable(self, own_pos, friend_pos, goal_pos):
        return lambda pos: (self.cfg["defendable_base_score"] \
                + self.cfg["defendable_distance_weight"] * vec_dist(pos, own_pos) \
                + self.cfg["defendable_friend_distance_weight"] * vec_dist(pos, friend_pos) \
                + self.cfg["defendable_friend_distance_weight"] * vec_dist(pos, goal_pos)  \
                + smoothstep(self._min_dist_to_neutral_corner(pos) * self.cfg["corner_safety_scale"]) * self.cfg["corner_safety_weight"])\
                if pos else -10000

    # --- Main policy ------------------------------------------------------------
    def get_controls(
        self,
        bot_coords: Sequence[Tuple[Tuple[float, float], float]],
        green_coords: Sequence[Tuple[float, float]],
        red_coords: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        own_pos = bot_coords[self._index][0]
        friend_pos = bot_coords[self._get_friend_index()][0]
        own_heading = bot_coords[self._index][1]

        base = self._own_base()
        dist_to_base = vec_dist(base, own_pos)
        opponent_goal = (1.5 - base[0], 1.5 - base[1])

        # Hysteresis: if we're "on our way home", expand the defend zone a bit;
        # this reduces rapid switching when a ball is near the frontier.
        defend_margin = self.cfg["hysteresis"] if self._going_to_base else 0.0

        # Partition balls by whether they lie beyond or inside our base frontier.
        attackables = [p for p in red_coords if vec_dist(base, p) > dist_to_base]
        defendables = [p for p in green_coords if vec_dist(base, p) < dist_to_base - defend_margin]

        # Default target is a calm hold position (mirrored to the other diagonal
        # if the board is empty).
        target = self._wait_position()
        if len(red_coords) == 0:
            target = (target[1], target[0])

        if not attackables and not defendables:
            if red_coords:
                self._going_to_base = True
                target = base
            elif green_coords:
                self._going_to_base = False
                target = opponent_goal
            return steer_to_target(own_pos, own_heading, target)

        best_attackable = max(attackables, key = self.score_attackable(own_pos, friend_pos, opponent_goal)) if attackables else None
        best_defendable = max(defendables, key = self.score_defendable(own_pos, friend_pos, base)) if defendables else None
        
        choose_attackable = self.score_attackable(own_pos, friend_pos, opponent_goal)(best_attackable) \
                > self.score_defendable(own_pos, friend_pos, base)(best_defendable)

        # Choose target + update state flag
        if choose_attackable:
            target = best_attackable
            self._going_to_base = False
        else:
            target = best_defendable
            self._going_to_base = True

        # Convert target into differential wheel commands.
        return steer_to_target(own_pos, own_heading, target)

