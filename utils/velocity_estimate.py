from .vec2d import *
from collections import namedtuple
from typing import List
from .math_utils import project_on_line_segment

MovingPoint = namedtuple("MovingPoint", ['position', 'velocity'])
def predict_coords(points: List[MovingPoint]):
    return [MovingPoint(vec_add(point.position,  point.velocity), point.velocity) for point in points]

def velocity_estimate(old_points, new_coords, alpha=0.9):
    predicted_coords = predict_coords(old_points)
    paired_old = [min(predicted_coords, key = lambda old: vec_dist(old.position, new)) for new in new_coords]
    error = [vec_sub(u,p.position) for u,p in zip(new_coords, paired_old)]
    new_vel = [vec_add(p.velocity, e) for p, e in zip(paired_old, error)]
    filter_vel = [vec_mix(p.velocity, new_v, 1.0 - alpha) for p, new_v in zip(paired_old, new_vel)]
    new_points = [MovingPoint(p,v) for p,v in zip(new_coords, filter_vel)]
    return new_points

def _intercept_coords(bot_coords, vel, points: List[MovingPoint], arena_width, arena_height, times):
    points = [MovingPoint(vec_add(p.position,  vec_mul(p.velocity, t)), p.velocity) for p, t in zip(points, times)]
    for i in range(len(points)):
        x = points[i].position[0]
        y = points[i].position[1]
        vx = points[i].velocity[0]
        vy = points[i].velocity[1]
        if x < 0.0:
            x *= -1.0
            vx *= -1.0
        if y < 0.0:
            y *= -1.0
            vy *= -1.0
        if x > arena_width:
            x = 2.0*arena_width - x
            vx *= -1.0
        if y > arena_height:
            y = 2.0*arena_height - y
            vy *= -1.0
        points[i] = MovingPoint((x, y), (vx, vy))
    return points

def intercept_coords(bot_coords, vel, points: List[MovingPoint], arena_width, arena_height, iterations = 2):
    times = [vec_dist(bot_coords, p.position) / vel for p in points]
    for _ in range(iterations):
        ppoints = _intercept_coords(bot_coords, vel, points, arena_width, arena_height,times)
        times = [vec_dist(bot_coords, p.position) / vel for p in ppoints]
    return ppoints

class Predictor:
    def __init__(self, index, iterations=2, max_vel = 0.001):
        self.red_cores = None
        self.green_cores = None
        self.bots = None
        self.index = index
        self.max_vel = max_vel
        self._iterations = iterations
    def observe(self, bot_coords, green_coords, red_coords):
        if self.red_cores == None:
            self.red_cores = [MovingPoint(x, (0.0, 0.0)) for x in red_coords]
        if self.green_cores == None:
            self.green_cores = [MovingPoint(x, (0.0, 0.0)) for x in green_coords]
        self.red_cores = velocity_estimate(self.red_cores, red_coords)
        self.green_cores = velocity_estimate(self.green_cores, green_coords)
        if self.bots == None:
            self.bots = [(x[0][0], x[0][1]) for x in bot_coords]
        else:
            vels = [vec_dist(a, b[0]) for a,b in zip(self.bots, bot_coords)]
            max_vel = max(vels)
            self.bots = [(x[0][0], x[0][1]) for x in bot_coords]
            if max_vel > self.max_vel:
                self.max_vel = max_vel
    def predict_red(self):
        ic = intercept_coords(self.bots[self.index],self.max_vel, self.red_cores, 1.5, 1.5, self._iterations)
        return [x.position for x in ic]
    def predict_green(self):
        ic = intercept_coords(self.bots[self.index],self.max_vel, self.green_cores, 1.5, 1.5, self._iterations)
        return [x.position for x in ic]