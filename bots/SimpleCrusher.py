from bots import register_bot
from bots.utility_functions import *

@register_bot
class SimpleCrusher:
    """
    A deliberately *SimpleBot-like* policy, but with:
      • Real behind-the-ball approach + push-through (no wiggle).
      • Deterministic split: even-index picks best candidate, odd-index the next best.
      • Hard lock (no chattering) + quick egress after entering any base.

    Target choice replicates SimpleBot’s intent:
      - Prefer REDs that are “farther from OUR base than we are” (offensive red play).
      - Else prefer GREENs that are “closer to OUR base than we are” (easy scores).
    """

    # Arena / base
    ARENA_W = 1.5
    ARENA_H = 1.5
    BASE_SIZE = 0.4

    # Controller
    TURN_GAIN = 1.55          # tuned to avoid in-place oscillation
    ALIGN_TURN_ONLY = 80.0    # deg; if worse than this -> rotate in place, no forward
    MAX_SPEED = 1.0

    # Ball handling
    APPROACH_OFFSET = 0.090   # stand-off behind the ball (m)
    CAPTURE_R = 0.060         # within this -> PUSH mode
    ALIGN_PUSH_DEG = 28.0     # must be roughly aligned with ball->goal to start PUSH

    # Locking / selection
    LOCK_TOL = 0.06           # same-ball tolerance (m)
    LOCK_MARGIN = 0.22        # only switch if clearly better (m)

    # Egress (leave any base fast to avoid getting stuck)
    EGRESS_DIST = 0.38        # step toward arena center (m)
    EGRESS_TICKS = 42

    # Light separation so mates don’t glue
    SEP_RADIUS = 0.18
    SEP_GAIN   = 0.08
    WALL_M     = 0.05

    # Stuck recovery
    HIST_N = 30
    STUCK_STEP_MIN = 0.0035
    ESCAPE_TICKS = 24

    # -------------- impl --------------
    def __init__(self, index):
        import math
        from collections import deque
        self._m = math
        self._index = index
        self._lock_kind = None     # 'red' or 'green'
        self._lock_pos = None
        self._push_mode = False
        self._egress_ticks = 0
        self._escape = 0
        self._hist = deque(maxlen=self.HIST_N)

    # --- tiny vec utils ---
    def _add(self,a,b): return (a[0]+b[0], a[1]+b[1])
    def _sub(self,a,b): return (a[0]-b[0], a[1]-b[1])
    def _mul(self,a,s): return (a[0]*s, a[1]*s)
    def _len(self,a): return self._m.hypot(a[0], a[1])
    def _norm(self,a):
        L = self._len(a)
        return (0.0,0.0) if L==0 else (a[0]/L, a[1]/L)
    def _perp(self,a): return (-a[1], a[0])
    def _dist(self,a,b): return self._len(self._sub(a,b))
    def _clamp(self,x,lo,hi): return lo if x<lo else hi if x>hi else x
    def _clamp_pt(self,p):
        return (self._clamp(p[0], self.WALL_M, self.ARENA_W - self.WALL_M),
                self._clamp(p[1], self.WALL_M, self.ARENA_H - self.WALL_M))

    # --- bases ---
    def _own_base(self): return get_base_coords(self._index)
    def _enemy_base(self):
        bx, by = self._own_base()
        corners = [(0.0,0.0),(self.ARENA_W,0.0),(0.0,self.ARENA_H),(self.ARENA_W,self.ARENA_H)]
        return max(corners, key=lambda c: self._dist(c, (bx,by)))

    def _base_axes(self, corner):
        cx, cy = (self.ARENA_W*0.5, self.ARENA_H*0.5)
        ix = 1.0 if corner[0] < cx else -1.0
        iy = 1.0 if corner[1] < cy else -1.0
        return (ix,0.0),(0.0,iy)

    def _base_center(self, corner):
        ax, ay = self._base_axes(corner)
        return self._add(self._add(corner, self._mul(ax, self.BASE_SIZE*0.5)), self._mul(ay, self.BASE_SIZE*0.5))

    def _base_rect(self, corner, margin=0.0):
        ax, ay = self._base_axes(corner)
        far = self._add(self._add(corner, self._mul(ax, self.BASE_SIZE)), self._mul(ay, self.BASE_SIZE))
        xmin, xmax = sorted([corner[0], far[0]])
        ymin, ymax = sorted([corner[1], far[1]])
        return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)

    # --- controller ---
    def _drive(self, own_pos, own_dir, target):
        dx, dy = target[0]-own_pos[0], target[1]-own_pos[1]
        if dx == 0.0 and dy == 0.0: return [0.0, 0.0]
        t_ang = self._m.atan2(dy, dx)
        err = (t_ang - own_dir + self._m.pi) % (2*self._m.pi) - self._m.pi
        if abs(err) > self._m.radians(self.ALIGN_TURN_ONLY):
            fwd = 0.0
        else:
            fwd = self.MAX_SPEED * max(0.0, self._m.cos(err))
        turn = self.TURN_GAIN * err
        l = fwd - turn; r = fwd + turn
        mag = max(1.0, abs(l), abs(r))
        return [l/mag, r/mag]

    # --- approach helpers ---
    def _approach_point(self, ball, goal_point):
        to_goal = self._norm(self._sub(goal_point, ball))
        if to_goal == (0.0,0.0): to_goal = (1.0,0.0)
        ap = self._add(ball, self._mul(to_goal, -self.APPROACH_OFFSET))
        return self._clamp_pt(ap), to_goal

    # --- small nudges (mate + walls) ---
    def _nudge(self, own_pos, bot_coords):
        n = (0.0,0.0)
        mate = (self._index ^ 1) if (self._index < 2) else (2 + ((self._index - 2) ^ 1))
        if 0 <= mate < len(bot_coords):
            mate_pos, _ = bot_coords[mate]
            d = self._dist(own_pos, mate_pos)
            if 1e-6 < d < self.SEP_RADIUS:
                away = self._norm(self._sub(own_pos, mate_pos))
                n = self._add(n, self._mul(away, self.SEP_GAIN * (self.SEP_RADIUS - d) / self.SEP_RADIUS))
        # gentle wall push
        x,y = own_pos
        if x < self.WALL_M: n = self._add(n, ( self.SEP_GAIN*(self.WALL_M-x)/self.WALL_M, 0.0))
        if self.ARENA_W - x < self.WALL_M: n = self._add(n, (-self.SEP_GAIN*(self.WALL_M-(self.ARENA_W-x))/self.WALL_M, 0.0))
        if y < self.WALL_M: n = self._add(n, (0.0,  self.SEP_GAIN*(self.WALL_M-y)/self.WALL_M))
        if self.ARENA_H - y < self.WALL_M: n = self._add(n, (0.0, -self.SEP_GAIN*(self.WALL_M-(self.ARENA_H-y))/self.WALL_M))
        return n

    # --- stuck & egress ---
    def _update_hist_and_check_stuck(self, own_pos):
        self._hist.append(own_pos)
        if len(self._hist) < self._hist.maxlen: return False
        dist = 0.0; prev = None
        for p in self._hist:
            if prev is not None: dist += self._dist(p, prev)
            prev = p
        return (dist / (len(self._hist)-1)) < self.STUCK_STEP_MIN

    def _escape_controls(self):
        s = -1.0 if (self._index % 2 == 0) else 1.0
        return [self._clamp(-0.55 - 0.30*s, -1, 1), self._clamp(-0.55 + 0.30*s, -1, 1)]

    def _start_egress_from(self, corner):
        self._egress_ticks = self.EGRESS_TICKS
        self._egress_from = corner

    def _egress_target(self, corner):
        center = self._base_center(corner)
        out = self._norm(self._sub((self.ARENA_W*0.5, self.ARENA_H*0.5), center))
        tgt = self._add(center, self._mul(out, self.EGRESS_DIST))
        return self._clamp_pt(tgt)

    # --- idling ---
    def getWaitPosition(self):
        return (0.48, 1.08) if self._index < 2 else (1.08, 0.48)

    # --- SimpleBot-like candidate ranking ---
    def _simplebot_candidates(self, own_pos, own_base, green_coords, red_coords):
        own_base_dist = self._dist(own_base, own_pos)
        reds = [r for r in red_coords if self._dist(own_base, r) > own_base_dist]
        greens = [g for g in green_coords if self._dist(own_base, g) < own_base_dist]

        # Rank by distance from us (like SimpleBot)
        reds_ranked = sorted(reds, key=lambda p: self._dist(own_pos, p))
        greens_ranked = sorted(greens, key=lambda p: self._dist(own_pos, p))
        return reds_ranked, greens_ranked

    # ============================ MAIN ============================
    def get_controls(self, bot_coords, green_coords, red_coords):
        own_pos, own_dir = bot_coords[self._index]
        own_base = self._own_base()
        enemy_base = self._enemy_base()
        own_center = self._base_center(own_base)

        # Escape if stuck
        if self._escape > 0:
            self._escape -= 1
            self._hist.clear()
            return self._escape_controls()
        if self._update_hist_and_check_stuck(own_pos):
            self._escape = self.ESCAPE_TICKS
            return self._escape_controls()

        # Egress if currently in any base
        for corner in (own_base, enemy_base):
            xmin, xmax, ymin, ymax = self._base_rect(corner, margin=0.0)
            if xmin <= own_pos[0] <= xmax and ymin <= own_pos[1] <= ymax:
                if self._egress_ticks == 0:
                    self._start_egress_from(corner)
                break
        if self._egress_ticks > 0:
            self._egress_ticks -= 1
            tgt = self._add(self._egress_target(self._egress_from), self._nudge(own_pos, bot_coords))
            return self._drive(own_pos, own_dir, tgt)

        # --- Candidate selection mirroring SimpleBot, but split between teammates ---
        reds_ranked, greens_ranked = self._simplebot_candidates(own_pos, own_base, green_coords, red_coords)

        choice_kind, choice_ball = None, None
        if reds_ranked:
            # offensive red: push into ENEMY base
            idx = 0 if (self._index % 2 == 0) else min(1, len(reds_ranked)-1)
            choice_kind, choice_ball = 'red', reds_ranked[idx]
        elif greens_ranked:
            # easy green near OUR base
            idx = 0 if (self._index % 2 == 0) else min(1, len(greens_ranked)-1)
            choice_kind, choice_ball = 'green', greens_ranked[idx]

        # Acquire / keep lock with hysteresis
        if self._lock_kind is None or self._lock_pos is None:
            if choice_ball is not None:
                self._lock_kind, self._lock_pos, self._push_mode = choice_kind, choice_ball, False
            else:
                # idle
                park = self._add(self.getWaitPosition(), self._nudge(own_pos, bot_coords))
                return self._drive(own_pos, own_dir, park)
        else:
            # refresh lock if same ball still present
            same = None
            balls = red_coords if self._lock_kind=='red' else green_coords
            for b in balls:
                if self._dist(b, self._lock_pos) <= self.LOCK_TOL:
                    same = b; break
            if same is not None:
                self._lock_pos = same
            elif choice_ball is not None:
                # switch only if clearly better
                cur_ap, _ = self._approach_point(self._lock_pos, enemy_base if self._lock_kind=='red' else own_center)
                new_ap, _ = self._approach_point(choice_ball, enemy_base if choice_kind=='red' else own_center)
                cur_cost = self._dist(own_pos, cur_ap)
                new_cost = self._dist(own_pos, new_ap)
                if new_cost + self.LOCK_MARGIN < cur_cost:
                    self._lock_kind, self._lock_pos, self._push_mode = choice_kind, choice_ball, False
            else:
                park = self._add(self.getWaitPosition(), self._nudge(own_pos, bot_coords))
                return self._drive(own_pos, own_dir, park)

        # --- Execute (approach -> push) ---
        goal_point = enemy_base if self._lock_kind == 'red' else own_center
        ap, to_goal = self._approach_point(self._lock_pos, goal_point)

        # enter push when close & aligned
        if not self._push_mode:
            desired = self._m.atan2(to_goal[1], to_goal[0])
            err = (desired - own_dir + self._m.pi) % (2*self._m.pi) - self._m.pi
            if (self._dist(own_pos, self._lock_pos) <= self.CAPTURE_R) and (abs(err) <= self._m.radians(self.ALIGN_PUSH_DEG)):
                self._push_mode = True

        # if inside our base while pushing, schedule egress (don’t camp)
        xmin, xmax, ymin, ymax = self._base_rect(own_base, 0.0)
        if self._push_mode and (xmin <= own_pos[0] <= xmax) and (ymin <= own_pos[1] <= ymax):
            if self._egress_ticks == 0:
                self._start_egress_from(own_base)

        target = goal_point if self._push_mode else ap
        target = self._add(target, self._nudge(own_pos, bot_coords))
        return self._drive(own_pos, own_dir, target)

