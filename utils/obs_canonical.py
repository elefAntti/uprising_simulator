
# utils/obs_canonical.py (fixed)
# Canonicalization + symmetry transforms with correct angle and base reflection.
import math
from typing import List, Tuple

Vec2 = Tuple[float, float]

def canonicalize_balls(balls: List[Vec2], goal_xy: Vec2) -> List[Vec2]:
    if not balls: return []
    def key(p):
        dx, dy = p[0]-goal_xy[0], p[1]-goal_xy[1]
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        return (round(d, 4), round(ang, 4))
    return sorted(balls, key=key)

def _wrap_pi(a: float) -> float:
    # Wrap angle to (-pi, pi]
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

def diagonal_reflect_point(p: Vec2, S: float) -> Vec2:
    # (x,y) -> (S - y, S - x)
    return (S - p[1], S - p[0])

def diagonal_reflect_angle(theta: float) -> float:
    # Under reflection (x,y)->(S-y,S-x), a heading vector v=[cos, sin] maps by M=[[0,-1],[-1,0]]: v' = (-sin, -cos).
    # That corresponds to theta' = -theta - pi/2 (mod 2pi).
    return _wrap_pi(-theta - math.pi/2.0)

def reflect_world(bot_coords, red_coords, green_coords, field_size: float):
    bots_r = [ (diagonal_reflect_point(p, field_size), diagonal_reflect_angle(ang)) for (p, ang) in bot_coords ]
    reds_r = [ diagonal_reflect_point(p, field_size) for p in red_coords ]
    greens_r = [ diagonal_reflect_point(p, field_size) for p in green_coords ]
    return bots_r, reds_r, greens_r

def reflect_bases(base_own: Vec2, base_opp: Vec2, field_size: float):
    return diagonal_reflect_point(base_own, field_size), diagonal_reflect_point(base_opp, field_size)
