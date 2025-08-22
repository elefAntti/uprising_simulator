
from collections import deque
import math
from bots.utility_functions import *
from utils.math_utils import * 
from bots import register_bot

ARENA_W, ARENA_H = 1.5, 1.5

def clamp(x, a, b): return a if x < a else b if x > b else x

@register_bot
class RedSlammer:
    """
    Crisp, low-chatter policy with optimizable parameters.

    Roles:
      - Even index (0,2): STRIKER -> prefers REDs toward enemy, then greens.
      - Odd  index (1,3): SWEEPER -> clears reds near own base; else supports.

    Tunables:
      See DEFAULTS/PARAM_SPEC for descriptions and ranges.
    """

    # -------- Optimization-friendly parameter sets --------
    DEFAULTS = {
        # Geometry / behavior
        "approach_offset": 0.11,      # m, behind-ball standoff
        "capture_r":       0.06,      # m, we "have" the ball within this radius
        "threat_r":        0.35,      # m, red considered a threat to our base
        "lock_margin":     0.20,      # m, new plan must beat current by this many meters of cost
        "lock_tol":        0.06,      # m, matching radius for a previously locked ball
        # Teammate & wall separation
        "sep_radius":      0.18,      # m
        "sep_gain":        0.10,      # unitless
        "wall_margin":     0.06,      # m, start nudging away from wall
        # Stuck detection
        "hist_n":          32,        # steps of history
        "stuck_step_min":  0.004,     # m per tick
        "escape_ticks":    28,        # ticks of escape maneuver
        # Drive (diff-drive smoother)
        "max_speed":       1.0,       # track command scale
        "turn_gain":       2.4,       # yaw gain
        "slow_angle_deg":  50,        # deg, brake when |err| > this
    }

    # Explicit ranges so CMA/GA can discover stronger configs
    PARAM_SPEC = {
        "approach_offset": {"lo": 0.08, "hi": 0.18, "type": "float"},
        "capture_r":       {"lo": 0.05, "hi": 0.08, "type": "float"},
        "threat_r":        {"lo": 0.25, "hi": 0.50, "type": "float"},
        "lock_margin":     {"lo": 0.10, "hi": 0.35, "type": "float"},
        "lock_tol":        {"lo": 0.04, "hi": 0.10, "type": "float"},

        "sep_radius":      {"lo": 0.12, "hi": 0.24, "type": "float"},
        "sep_gain":        {"lo": 0.05, "hi": 0.20, "type": "float"},
        "wall_margin":     {"lo": 0.04, "hi": 0.10, "type": "float"},

        "hist_n":          {"lo": 16, "hi": 64, "type": "int"},
        "stuck_step_min":  {"lo": 0.002, "hi": 0.010, "type": "float"},
        "escape_ticks":    {"lo": 16, "hi": 48, "type": "int"},

        "max_speed":       {"lo": 0.8, "hi": 1.2, "type": "float"},
        "turn_gain":       {"lo": 1.6, "hi": 3.2, "type": "float"},
        "slow_angle_deg":  {"lo": 30, "hi": 70, "type": "int"},
    }

    def __init__(self, index, **params):
        self._index = index
        self._role_striker = (index % 2 == 0)
        # Merge defaults with overrides
        cfg = dict(self.DEFAULTS); cfg.update(params or {})
        # Bind as attributes
        for k, v in cfg.items():
            setattr(self, k, v)
        # Precompute radians
        self.slow_angle = math.radians(float(self.slow_angle_deg))
        # Runtime state
        self._lock_kind = None   # 'red' | 'green' | None
        self._lock_pos = None
        self._lock_id = None
        self._push_mode = False  # when close to ball, target = goal (push-through)
        self._hist = deque(maxlen=int(self.hist_n))
        self._escape_ticks = 0

    # --- basics ---
    def _own_base(self):
        return get_base_coords(self._index)

    def _enemy_base(self):
        bx, by = self._own_base()
        corners = [(0.0, 0.0), (ARENA_W, 0.0), (0.0, ARENA_H), (ARENA_W, ARENA_H)]
        return max(corners, key=lambda c: vec_dist(c, (bx, by)))

    def getWaitPosition(self):
        # Simple safe loiter spots
        if self._index < 2:
            return (0.35 + 0.12*(self._index%2), 1.10)
        else:
            return (1.10, 0.35 + 0.12*(self._index%2))

    def _approach_point(self, ball, goal):
        """Point behind the ball along ball->goal direction."""
        bx, by = ball
        gx, gy = goal
        to_goal = self._vnorm((gx - bx, gy - by))
        if to_goal == (0.0, 0.0): to_goal = (1.0, 0.0)
        return (bx - to_goal[0]*self.approach_offset, by - to_goal[1]*self.approach_offset)

    def _id_of(self, pos):
        # quantize to centimeters for stability
        return (round(pos[0], 2), round(pos[1], 2))

    # --- tiny avoidance vs teammate & walls (kept small to avoid dithering) ---
    def _nudge(self, own_pos, bot_coords):
        ox, oy = own_pos
        n = (0.0, 0.0)
        # teammate only: xor 1 within team
        mate_idx = (self._index ^ 1) if (self._index < 2) else (2 + ((self._index - 2) ^ 1))
        if 0 <= mate_idx < len(bot_coords):
            mate_pos, _ = bot_coords[mate_idx]
            d = vec_dist(own_pos, mate_pos)
            if d < self.sep_radius and d > 1e-6:
                away = self._vnorm(self._vsub(own_pos, mate_pos))
                n = self._vadd(n, self._vmul(away, self.sep_gain * (self.sep_radius - d) / self.sep_radius))
        # gentle wall nudge
        M = self.wall_margin
        left = ox; right = ARENA_W - ox; bottom = oy; top = ARENA_H - oy
        if left < M:  n = self._vadd(n, (self.sep_gain*(M-left)/M, 0.0))
        if right < M: n = self._vadd(n, (-self.sep_gain*(M-right)/M, 0.0))
        if bottom < M:n = self._vadd(n, (0.0, self.sep_gain*(M-bottom)/M))
        if top < M:   n = self._vadd(n, (0.0, -self.sep_gain*(M-top)/M))
        return n

    # --- costs (lower is better) ---
    def _approach_cost(self, own, ball, goal):
        """Distance to approach point + weighted ball->goal distance."""
        ap = self._approach_point(ball, goal)
        return vec_dist(own, ap) + 0.7 * vec_dist(ball, goal)

    def _best_red(self, own, reds, enemy_base):
        if not reds: return None
        best = min(reds, key=lambda r: self._approach_cost(own, r, enemy_base))
        return ('red', best)

    def _best_green(self, own, greens, own_base):
        if not greens: return None
        best = min(greens, key=lambda g: self._approach_cost(own, g, own_base))
        return ('green', best)

    # --- lock logic ---
    def _match_locked_ball(self, balls):
        """Return updated position of the locked ball if it still exists nearby."""
        if self._lock_pos is None: return None
        tol = self.lock_tol
        near = [b for b in balls if vec_dist(b, self._lock_pos) <= tol]
        if not near: return None
        return min(near, key=lambda b: vec_dist(b, self._lock_pos))

    # --- stuck detection ---
    def _update_hist_and_check_stuck(self, own_pos):
        self._hist.append(own_pos)
        if len(self._hist) < self._hist.maxlen: return False
        dist = 0.0
        prev = None
        for p in self._hist:
            if prev is not None:
                dist += vec_dist(p, prev)
            prev = p
        avg_step = dist / (len(self._hist)-1)
        return avg_step < self.stuck_step_min

    def _escape_controls(self):
        # short reverse + slight spin; even/odd spin opposite
        s = -1.0 if (self._index % 2 == 0) else 1.0
        left = clamp(-0.5 - 0.35*s, -1, 1)
        right = clamp(-0.5 + 0.35*s, -1, 1)
        return [left, right]

    # --- vector helpers ---
    @staticmethod
    def _vlen(a): return math.hypot(a[0], a[1])
    @staticmethod
    def _vsub(a,b): return (a[0]-b[0], a[1]-b[1])
    @staticmethod
    def _vadd(a,b): return (a[0]+b[0], a[1]+b[1])
    @staticmethod
    def _vmul(a,s): return (a[0]*s, a[1]*s)
    @staticmethod
    def _vnorm(a):
        L = math.hypot(a[0], a[1])
        return (0.0,0.0) if L==0 else (a[0]/L, a[1]/L)

    # --- drive ---
    def _smooth_drive_to_target(self, own_pos, own_dir, target):
        """Smooth diff-drive aimed at *not* oscillating. Uses tuned gains."""
        tx, ty = target
        ox, oy = own_pos
        dx, dy = tx - ox, ty - oy
        if dx == 0.0 and dy == 0.0:
            return [0.0, 0.0]
        target_angle = math.atan2(dy, dx)
        err = (target_angle - own_dir + math.pi) % (2*math.pi) - math.pi

        align = max(0.0, math.cos(err))
        if abs(err) > self.slow_angle:  # brake while turning hard
            align *= 0.45

        fwd = self.max_speed * align
        turn = self.turn_gain * err

        left = fwd - turn
        right = fwd + turn
        m = max(1.0, abs(left), abs(right))
        return [left / m, right / m]

    # --- main ---
    def get_controls(self, bot_coords, green_coords, red_coords):
        own_pos, own_dir = bot_coords[self._index]
        own_base = self._own_base()
        enemy_base = self._enemy_base()

        # Escape if stuck
        if self._escape_ticks > 0:
            self._escape_ticks -= 1
            self._hist.clear()
            return self._escape_controls()
        if self._update_hist_and_check_stuck(own_pos):
            self._escape_ticks = int(self.escape_ticks)
            return self._escape_controls()

        # 1) Sweeper: clear red threats near our base
        if not self._role_striker:
            threats = [r for r in red_coords if vec_dist(own_base, r) <= self.threat_r]
            if threats:
                ball = min(threats, key=lambda r: vec_dist(own_base, r))
                self._lock_kind, self._lock_pos, self._lock_id = 'red', ball, self._id_of(ball)
                self._push_mode = False

        # 2) Else pick target based on role (with lock-on)
        if self._lock_kind is None:
            choice = None
            if self._role_striker:
                choice = self._best_red(own_pos, red_coords, enemy_base) or self._best_green(own_pos, green_coords, own_base)
            else:
                nearby_reds = [r for r in red_coords if vec_dist(own_pos, r) < 0.8]
                choice = self._best_red(own_pos, nearby_reds, enemy_base)                          or self._best_red(own_pos, red_coords, enemy_base)                          or self._best_green(own_pos, green_coords, own_base)

            if choice:
                k, b = choice
                self._lock_kind, self._lock_pos, self._lock_id = k, b, self._id_of(b)
                self._push_mode = False
        else:
            # keep lock if ball still around
            locked_now = self._match_locked_ball(red_coords if self._lock_kind=='red' else green_coords)
            if locked_now is not None:
                self._lock_pos = locked_now
            else:
                self._lock_kind = None
                self._lock_pos = None
                self._lock_id = None
                self._push_mode = False

        # 3) If we have a target, drive a straight plan with tiny nudges
        if self._lock_kind and self._lock_pos is not None:
            goal = enemy_base if self._lock_kind == 'red' else own_base

            # Enter push mode when close to ball
            if not self._push_mode and vec_dist(own_pos, self._lock_pos) <= self.capture_r:
                self._push_mode = True

            # Decide whether to keep the lock vs a better red (aggressive striker)
            if self._role_striker and self._lock_kind == 'green' and red_coords:
                best_red = self._best_red(own_pos, red_coords, enemy_base)
                if best_red:
                    _, red_ball = best_red
                    new_cost = self._approach_cost(own_pos, red_ball, enemy_base)
                    cur_cost = self._approach_cost(own_pos, self._lock_pos, goal)
                    if new_cost + self.lock_margin < cur_cost:
                        self._lock_kind, self._lock_pos, self._lock_id = 'red', red_ball, self._id_of(red_ball)
                        self._push_mode = False
                        goal = enemy_base

            # Target point
            target = goal if self._push_mode else self._approach_point(self._lock_pos, goal)

            # Tiny separation/wall nudge to avoid sticking — but keep it VERY small
            target = self._vadd(target, self._nudge(own_pos, bot_coords))

            return self._smooth_drive_to_target(own_pos, own_dir, target)

        # 4) No target: go park (with tiny nudge so we don’t hug walls)
        park = self._vadd(self.getWaitPosition(), self._nudge(own_pos, bot_coords))
        return self._smooth_drive_to_target(own_pos, own_dir, park)
