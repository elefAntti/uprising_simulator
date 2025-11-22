
"""
utils/ball_tracker.py
---------------------
Multi-ball tracker for overhead camera measurements.

Features
--------
- Stable IDs across frames via **data association** (Hungarian if available, greedy fallback).
- Per-ball **constant-velocity Kalman filter** (state = [x, y, vx, vy]).
- Handles births (new balls), deaths (balls removed), missed detections.
- Provides **velocity estimates** and short-term prediction (predict next position).
- Uses **Mahalanobis gating** (2x2 S inverse) to avoid bad matches.
- No heavy dependencies; optionally uses SciPy if installed.

Typical usage
-------------
>>> from utils.ball_tracker import BallTracker
>>> bt = BallTracker(gate_sigma=3.0)
>>> tracks = bt.update([(0.2, 0.3), (1.1, 1.2)], timestamp=0.00)
>>> tracks = bt.update([(0.21, 0.32), (1.08, 1.18)], timestamp=0.05)
>>> for tr in tracks:
...     print(tr['id'], tr['pos'], tr['vel'])

Integration pattern
-------------------
- Call `update(measurements, timestamp=...)` each camera frame.
- For bots that need **predicted** positions for the next control step, use
  `pred = tracker.predict_positions(dt_ahead)`.
- To pass stable "red_coords" to bots, use `tracker.get_positions()` which preserves ID order
  (sorted by track id) so your "index" never relies on raw measurement order.

Notes
-----
- Measurement is 2D (x, y) in meters in field coordinates.
- If your frame time delta is unknown, pass `dt=...` explicitly.
- Tuning: `process_var` ≈ expected acceleration variance; `meas_var` ≈ camera noise variance.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

try:
    # Optional: if SciPy is around, prefer optimal assignment
    from scipy.optimize import linear_sum_assignment as _hungarian
    _HAVE_SCIPY = True
except Exception:
    _hungarian = None
    _HAVE_SCIPY = False

Vec2 = Tuple[float, float]

# ------------- Small linear-algebra helpers (2x2 only) -------------
def _inv2(a11, a12, a21, a22):
    det = a11*a22 - a12*a21
    if abs(det) < 1e-12:
        # pseudo-inverse fallback
        det = 1e-12
    inv = ( a22/det, -a12/det,
           -a21/det,  a11/det )
    return inv

def _mahalanobis2(dx: Vec2, S: Tuple[float, float, float, float]) -> float:
    # S is 2x2 (s11, s12, s21, s22)
    inv = _inv2(*S)
    # y^T S^{-1} y
    return math.sqrt(dx[0]*(inv[0]*dx[0] + inv[1]*dx[1]) +
                     dx[1]*(inv[2]*dx[0] + inv[3]*dx[1]))

# ------------- Track (Kalman filter) -------------
class _KFTrack:
    __slots__ = ("id", "x", "P", "age", "hits", "misses", "last_ts")
    def __init__(self, tid: int, pos: Vec2, meas_var: float, v0_var: float, ts: float):
        # State: [x, y, vx, vy]
        self.id = tid
        self.x = [pos[0], pos[1], 0.0, 0.0]
        # Initial covariance: position ~ meas_var, velocity ~ v0_var
        self.P = [
            [meas_var,   0.0,      0.0,     0.0],
            [0.0,      meas_var,   0.0,     0.0],
            [0.0,        0.0,    v0_var,    0.0],
            [0.0,        0.0,      0.0,   v0_var],
        ]
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.last_ts = ts

    def predict(self, dt: float, q: float):
        # F
        F = [
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        # x = F x
        x0, y0, vx0, vy0 = self.x
        self.x = [x0 + dt*vx0, y0 + dt*vy0, vx0, vy0]

        # Q (CV model)
        dt2 = dt*dt; dt3 = dt2*dt
        q11 = dt3/3.0; q13 = dt2/2.0
        q22 = dt3/3.0; q24 = dt2/2.0
        Q = [
            [q*q11,     0.0,     q*q13,    0.0],
            [0.0,     q*q22,     0.0,    q*q24],
            [q*q13,     0.0,       q*dt2,  0.0],
            [0.0,     q*q24,     0.0,      q*dt2],
        ]

        # P = F P F^T + Q
        P = self.P
        # compute A = F P
        A = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for k in range(4):
                A[i][0] += F[i][k] * P[k][0]
                A[i][1] += F[i][k] * P[k][1]
                A[i][2] += F[i][k] * P[k][2]
                A[i][3] += F[i][k] * P[k][3]
        # P' = A F^T
        PF = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                PF[i][j] = A[i][0]*F[j][0] + A[i][1]*F[j][1] + A[i][2]*F[j][2] + A[i][3]*F[j][3]
        # add Q
        for i in range(4):
            for j in range(4):
                PF[i][j] += Q[i][j]
        self.P = PF

    def innovation(self, z: Vec2, meas_var: float):
        # H = [ [1,0,0,0], [0,1,0,0] ]
        hx = (self.x[0], self.x[1])
        y = (z[0] - hx[0], z[1] - hx[1])  # residual
        # S = H P H^T + R  (2x2)
        s11 = self.P[0][0] + meas_var
        s12 = self.P[0][1]
        s21 = self.P[1][0]
        s22 = self.P[1][1] + meas_var
        return y, (s11, s12, s21, s22)

    def update(self, z: Vec2, meas_var: float):
        # Innovation
        y, (s11, s12, s21, s22) = self.innovation(z, meas_var)
        # Kalman gain K = P H^T S^{-1}; but with H selecting first 2 rows/cols
        inv = _inv2(s11, s12, s21, s22)
        # K is 4x2:
        # K[:,0] = [P00, P10, P20, P30] * inv_col0 + [P01, P11, P21, P31] * inv_col1 ?
        # Explicit 4x2 product: K = P H^T S^{-1} where H^T selects first two columns.
        P = self.P
        # Compute columns of K
        k0 = [P[0][0]*inv[0] + P[0][1]*inv[2],
              P[1][0]*inv[0] + P[1][1]*inv[2],
              P[2][0]*inv[0] + P[2][1]*inv[2],
              P[3][0]*inv[0] + P[3][1]*inv[2]]
        k1 = [P[0][0]*inv[1] + P[0][1]*inv[3],
              P[1][0]*inv[1] + P[1][1]*inv[3],
              P[2][0]*inv[1] + P[2][1]*inv[3],
              P[3][0]*inv[1] + P[3][1]*inv[3]]
        # x = x + K y
        self.x[0] += k0[0]*y[0] + k1[0]*y[1]
        self.x[1] += k0[1]*y[0] + k1[1]*y[1]
        self.x[2] += k0[2]*y[0] + k1[2]*y[1]
        self.x[3] += k0[3]*y[0] + k1[3]*y[1]
        # P = (I - K H) P; with H selecting first two states
        # Build KH
        KH = [
            [k0[0], k1[0], 0.0, 0.0],
            [k0[1], k1[1], 0.0, 0.0],
            [k0[2], k1[2], 0.0, 0.0],
            [k0[3], k1[3], 0.0, 0.0],
        ]
        I = [[1.0 if i==j else 0.0 for j in range(4)] for i in range(4)]
        IMKH = [[I[i][j] - KH[i][j] for j in range(4)] for i in range(4)]
        # P = (I-KH) P
        newP = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                newP[i][j] = IMKH[i][0]*P[0][j] + IMKH[i][1]*P[1][j] + IMKH[i][2]*P[2][j] + IMKH[i][3]*P[3][j]
        self.P = newP
        self.hits += 1
        self.misses = 0

    def pos(self) -> Vec2:
        return (self.x[0], self.x[1])

    def vel(self) -> Vec2:
        return (self.x[2], self.x[3])

# ------------- Tracker -------------
class BallTracker:
    def __init__(self,
                 process_var: float = 1.0,       # accel variance (q)
                 meas_var: float = 1e-3,         # camera noise variance (r) ~ (meters^2)
                 v0_var: float = 1.0,            # initial velocity variance
                 gate_sigma: float = 3.0,        # gating in sigma (Mahalanobis)
                 max_missed: int = 5,            # drop track after this many misses
                 max_dt: float = 0.25,           # clamp dt to avoid huge predictions
                 use_hungarian: bool = True):
        self.q = float(process_var)
        self.r = float(meas_var)
        self.v0_var = float(v0_var)
        self.gate_k = float(gate_sigma)
        self.max_missed = int(max_missed)
        self.max_dt = float(max_dt)
        self.use_hungarian = bool(use_hungarian and _HAVE_SCIPY)
        self._tracks: List[_KFTrack] = []
        self._next_id = 1
        self._last_ts: Optional[float] = None

    # ---- Public API ----
    def update(self, measurements: List[Vec2], timestamp: Optional[float] = None, dt: Optional[float] = None) -> List[Dict]:
        """
        Update tracker with a new list of (x, y) measurements (meters).

        Either pass `timestamp` (seconds) to infer dt, or pass `dt` directly.
        Returns a list of track dicts: {'id', 'pos', 'vel', 'age', 'hits', 'misses'} sorted by id.
        """
        # Compute dt
        if timestamp is not None:
            if self._last_ts is None:
                dt_eff = dt if dt is not None else 0.05
            else:
                dt_eff = timestamp - self._last_ts if dt is None else dt
            self._last_ts = timestamp
        else:
            dt_eff = dt if dt is not None else 0.05

        dt_eff = max(1e-3, min(self.max_dt, dt_eff))

        # 1) Predict existing tracks
        for tr in self._tracks:
            tr.predict(dt_eff, self.q)

        # 2) Build cost matrix with gating
        M = len(self._tracks)
        N = len(measurements)
        if M == 0 and N > 0:
            # spawn new for all
            for z in measurements:
                self._spawn(z, timestamp or 0.0)
            return self.get_tracks()

        if M == 0:
            return self.get_tracks()

        BIG = 1e6
        cost = [[BIG]*N for _ in range(M)]
        gated = [[True]*N for _ in range(M)]
        for i, tr in enumerate(self._tracks):
            for j, z in enumerate(measurements):
                y, S = tr.innovation(z, self.r)
                d = _mahalanobis2(y, S)
                if d <= self.gate_k:
                    # cost as Euclidean distance (or use d), we keep Mahalanobis for stronger gating
                    eu = math.hypot(y[0], y[1])
                    cost[i][j] = eu
                    gated[i][j] = False

        # 3) Assignment
        matches = []
        unmatched_tracks = set(range(M))
        unmatched_meas = set(range(N))

        if self.use_hungarian and M>0 and N>0:
            row_ind, col_ind = _hungarian(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i][j] >= BIG or gated[i][j]:
                    continue
                matches.append((i, j))
        else:
            # Greedy: repeatedly pick the smallest cost
            used_tracks = set(); used_meas = set()
            pairs = []
            for i in range(M):
                for j in range(N):
                    if cost[i][j] < BIG:
                        pairs.append((cost[i][j], i, j))
            pairs.sort()
            for _, i, j in pairs:
                if i in used_tracks or j in used_meas:
                    continue
                matches.append((i, j))
                used_tracks.add(i); used_meas.add(j)

        # Update sets
        for i, j in matches:
            unmatched_tracks.discard(i)
            unmatched_meas.discard(j)

        # 4) Update matched tracks
        for i, j in matches:
            self._tracks[i].update(measurements[j], self.r)
            self._tracks[i].age += 1
            self._tracks[i].last_ts = timestamp if timestamp is not None else (self._tracks[i].last_ts + dt_eff)

        # 5) Handle unmatched tracks (missed)
        for i in list(unmatched_tracks):
            tr = self._tracks[i]
            tr.misses += 1
            tr.age += 1
            tr.last_ts = timestamp if timestamp is not None else (tr.last_ts + dt_eff)

        # 6) Spawn new tracks for unmatched measurements
        for j in unmatched_meas:
            self._spawn(measurements[j], timestamp or 0.0)

        # 7) Prune old tracks
        self._tracks = [t for t in self._tracks if t.misses <= self.max_missed]

        # 8) Return summary
        return self.get_tracks()

    def get_tracks(self) -> List[Dict]:
        out = []
        for tr in sorted(self._tracks, key=lambda t: t.id):
            out.append({
                "id": tr.id,
                "pos": (tr.x[0], tr.x[1]),
                "vel": (tr.x[2], tr.x[3]),
                "age": tr.age,
                "hits": tr.hits,
                "misses": tr.misses,
            })
        return out

    def get_positions(self) -> List[Vec2]:
        """Stable-order positions (sorted by track id)."""
        return [(t.x[0], t.x[1]) for t in sorted(self._tracks, key=lambda t: t.id)]

    def predict_positions(self, dt_ahead: float) -> List[Vec2]:
        """One-step prediction ahead (no covariance propagation returned)."""
        preds = []
        for tr in sorted(self._tracks, key=lambda t: t.id):
            x = tr.x
            preds.append((x[0] + x[2]*dt_ahead, x[1] + x[3]*dt_ahead))
        return preds

    def _spawn(self, z: Vec2, ts: float):
        tr = _KFTrack(self._next_id, z, self.r, self.v0_var, ts)
        self._tracks.append(tr)
        self._next_id += 1

# Convenience alias
Tracker = BallTracker

if __name__ == "__main__":
    # Tiny self-test
    bt = BallTracker(process_var=0.8, meas_var=1e-3, gate_sigma=3.0)
    t = 0.0
    seq = [
        [(0.2, 0.3), (1.1, 1.2)],
        [(0.25, 0.33), (1.06, 1.16)],
        [(0.28, 0.37), (1.02, 1.12)],
        [(0.32, 0.41)],               # one disappears
        [(0.36, 0.46), (1.00, 1.08)], # reappears (new id)
    ]
    for meas in seq:
        t += 0.05
        tracks = bt.update(meas, timestamp=t)
        print("t=%.2f" % t, " -> ", [(tr['id'], tr['pos'], tr['vel']) for tr in tracks])
