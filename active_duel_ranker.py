
"""
active_duel_ranker.py (improved)
--------------------------------
- Adds tqdm progress bar and live status ("undecided pairs").
- Noisy, non-transitive dueling ranker with Wilson intervals.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ------------------------------ Stats utils ----------------------------------
def wilson_interval(wins: float, n: int, z: float = 2.0) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * ((p * (1 - p) + z2 / (4 * n)) / n) ** 0.5 / denom
    lo = max(0.0, min(1.0, center - half))
    hi = max(0.0, min(1.0, center + half))
    return lo, hi

def topological_sort(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None

def strongly_connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    index = 0
    stack: List[int] = []
    onstack = [False] * n
    idx = [-1] * n
    low = [0] * n
    sccs: List[List[int]] = []

    def dfs(v: int):
        nonlocal index
        idx[v] = low[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True
        for w in adj[v]:
            if idx[w] == -1:
                dfs(w)
                low[v] = min(low[v], low[w])
            elif onstack[w]:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if idx[v] == -1:
            dfs(v)
    return sccs

DuelFn = Callable[[int, int, int], Tuple[int, int, int]]

@dataclass
class ActiveDuelRanker:
    items: Sequence[str]
    duel: DuelFn
    batch_size: int = 3
    z: float = 2.0
    strategy: str = "copeland-ucb"  # or "uncertainty"

    def __post_init__(self):
        n = len(self.items)
        self.n = n
        self.wins = [[0 for _ in range(n)] for _ in range(n)]
        self.draws = [[0 for _ in range(n)] for _ in range(n)]
        self.games = [[0 for _ in range(n)] for _ in range(n)]

    # ----- Pair stats ----------------------------------------------------------
    def p_hat(self, i: int, j: int) -> float:
        g = self.games[i][j]
        if g == 0:
            return 0.5
        w = self.wins[i][j] + 0.5 * self.draws[i][j]
        return w / g

    def ci(self, i: int, j: int) -> Tuple[float, float]:
        g = self.games[i][j]
        w = self.wins[i][j] + 0.5 * self.draws[i][j]
        return wilson_interval(w, g, self.z)

    def status(self, i: int, j: int) -> str:
        lo, hi = self.ci(i, j)
        if lo > 0.5:
            return "i>j"
        if hi < 0.5:
            return "j>i"
        return "undecided"

    def undecided_pairs_count(self) -> int:
        c = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.status(i, j) == "undecided":
                    c += 1
        return c

    # ----- Copeland optimistic/pessimistic bounds -----------------------------
    def copeland_bounds(self) -> Tuple[List[int], List[int]]:
        L = [0] * self.n
        U = [0] * self.n
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                lo, hi = self.ci(i, j)
                if lo > 0.5:
                    L[i] += 1
                if hi > 0.5:
                    U[i] += 1
        return L, U

    # ----- Next pair selection -------------------------------------------------
    def next_pair(self) -> Tuple[int, int]:
        undecided_pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.status(i, j) == "undecided":
                    lo, hi = self.ci(i, j)
                    width = hi - lo
                    center = (lo + hi) / 2.0
                    closeness = 1.0 - 2.0 * abs(center - 0.5)
                    undecided_pairs.append((i, j, width, closeness))
        if not undecided_pairs:
            return (-1, -1)

        if self.strategy == "uncertainty":
            undecided_pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
            return undecided_pairs[0][0], undecided_pairs[0][1]

        L, U = self.copeland_bounds()
        leader = max(range(self.n), key=lambda i: (U[i], L[i]))
        touching = [t for t in undecided_pairs if t[0] == leader or t[1] == leader]
        if touching:
            touching.sort(key=lambda x: (x[2], x[3]), reverse=True)
            return touching[0][0], touching[0][1]

        undecided_pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
        return undecided_pairs[0][0], undecided_pairs[0][1]

    # ----- Update from matches -------------------------------------------------
    def play(self, i: int, j: int, n_games: Optional[int] = None):
        if n_games is None:
            n_games = self.batch_size
        wi, wj, dr = self.duel(i, j, n_games)
        self.wins[i][j] += wi
        self.wins[j][i] += wj
        self.draws[i][j] += dr
        self.draws[j][i] += dr
        self.games[i][j] += (wi + wj + dr)
        self.games[j][i] += (wi + wj + dr)

    # ----- Graph construction --------------------------------------------------
    def edges(self) -> List[Tuple[int, int]]:
        E = []
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                lo, _ = self.ci(i, j)
                if lo > 0.5:
                    E.append((i, j))
        return E

    def partial_order(self) -> Tuple[Optional[List[int]], List[List[int]]]:
        E = self.edges()
        order = topological_sort(self.n, E)
        sccs = strongly_connected_components(self.n, E)
        cycles = [sorted(c) for c in sccs if len(c) >= 2]
        return order, cycles

    # ----- Stopping tests ------------------------------------------------------
    def is_total_order_certain(self) -> bool:
        order, _ = self.partial_order()
        if order is None:
            return False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.status(i, j) == "undecided":
                    return False
        return True

    def is_topk_certain(self, k: int) -> bool:
        L, U = self.copeland_bounds()
        idx = list(range(self.n))
        idx.sort(key=lambda i: (-L[i], -U[i], i))
        top = idx[:k]
        rest = idx[k:]
        return all(L[i] > U[j] for i in top for j in rest)

    # ----- Main loop -----------------------------------------------------------
    def run(
        self,
        max_matches: int = 10_000,
        stop_when_total_order: bool = True,
        topk: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, object]:
        played = 0
        bar = None
        if progress and tqdm is not None:
            bar = tqdm(total=max_matches, dynamic_ncols=True, leave=True)
            bar.set_description("Scheduling duels")
            bar.set_postfix_str("initializing...")

        while played < max_matches:
            if stop_when_total_order and self.is_total_order_certain():
                if bar: bar.set_postfix_str("total order proven")
                break
            if topk is not None and self.is_topk_certain(topk):
                if bar: bar.set_postfix_str(f"top-{topk} proven")
                break

            i, j = self.next_pair()
            if i < 0:
                if bar: bar.set_postfix_str("all pairs decided")
                break  # nothing left to decide

            if bar:
                bar.set_description(f"{self.items[i]} vs {self.items[j]}")
                bar.set_postfix_str(f"undecided={self.undecided_pairs_count()}")

            self.play(i, j)
            played += self.batch_size
            if bar:
                bar.update(self.batch_size)

        if bar:
            bar.close()

        order, cycles = self.partial_order()
        L, U = self.copeland_bounds()

        matrix_p = [[0.5] * self.n for _ in range(self.n)]
        matrix_lo = [[0.0] * self.n for _ in range(self.n)]
        matrix_hi = [[1.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                matrix_p[i][j] = self.p_hat(i, j)
                matrix_lo[i][j], matrix_hi[i][j] = self.ci(i, j)

        return {
            "played": played,
            "order": [self.items[i] for i in order] if order is not None else None,
            "cycles": [[self.items[i] for i in comp] for comp in cycles],
            "bounds": {"L": L, "U": U},
            "matrix": {"p": matrix_p, "lo": matrix_lo, "hi": matrix_hi},
        }
