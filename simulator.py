from collections import defaultdict
from game_data import *
import Box2D  # The main library
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, dot)
from utils.vec2d import *
import math
import random

# --- Randomness helpers for Monte Carlo runs ---
def _rand_mult(rel_sigma=0.1, lo_mult=0.7, hi_mult=1.3):
    """Sample a multiplicative factor ~ N(1, rel_sigma) and clamp to [lo_mult, hi_mult]."""
    f = random.gauss(1.0, rel_sigma)
    if f < lo_mult:
        f = lo_mult
    if f > hi_mult:
        f = hi_mult
    return f

def _jitter(base, *, rel_sigma=0.1, min_mult=0.7, max_mult=1.3, min_val=None, max_val=None):
    """Return base * factor with truncated Gaussian multiplicative noise.
    Optionally clamp absolute value to [min_val, max_val].
    """
    val = base * _rand_mult(rel_sigma=rel_sigma, lo_mult=min_mult, hi_mult=max_mult)
    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return val
# --- end randomness helpers ---


def categorize(func, seq):
    """Return mapping from categories to lists
    of categorized items.
    """
    d = defaultdict(list)
    for item in seq:
        d[func(item)].append(item)
    return d

def goal_state(core):
    det = vec_dot((1.0, -1.0), vec_sub(core.position, (0.0, ARENA_HEIGHT)))
    if det < BASE_SIZE:
        return 1
    if det > (ARENA_HEIGHT + ARENA_WIDTH - BASE_SIZE):
        return 2
    return 0

def steer_point(body, point, speed):
    forward = body.GetWorldVector((1,0))
    velocity = body.GetLinearVelocityFromWorldPoint(point)
    currentSpeed = dot(velocity, forward)
    err = currentSpeed - speed
    k = -7.0
    body.ApplyForce(forward * min(STEER_FORCE, max(-STEER_FORCE, k*err)), point, True)

def steer(body, left_speed, right_speed):
    steer_point(body, body.GetWorldPoint((0.0,0.05)), left_speed)
    steer_point(body, body.GetWorldPoint((0.0,-0.05)), right_speed)

    side = body.GetWorldVector((0,1))
    transSpeed = dot(body.linearVelocity, side)
    impulse=-transSpeed*body.mass*side
    body.ApplyLinearImpulse(impulse=impulse, point=body.position, wake=False)

def bvec_to_tuple(vec):
    return (vec[0], vec[1])

def GetRandomPoints(count, safe_distance = CORE_RADIUS):
    generated_points=[]
    min_val = safe_distance
    max_val = ARENA_WIDTH - safe_distance
    while len(generated_points) < count:
        point = (random.uniform(min_val, max_val),
                 random.uniform(min_val, max_val))
        if vec_dist(point, TEAM1_BASE) < BASE_SAFETY_AREA:
            continue
        if vec_dist(point, TEAM2_BASE) < BASE_SAFETY_AREA:
            continue
        if min((vec_dist(point, other) for other in generated_points),\
               default = float("inf")) > safe_distance * 2.0:
            generated_points.append(point)
    return generated_points

class Simulator:
    def init(self, controllers, randomize=False, noise=None, seed=None):
        self.controllers = controllers
        self.world = world(gravity=(0, 0), doSleep=True)
        self.walls = [
            self.world.CreateStaticBody(
                position=(0, -MARGIN),
                shapes=polygonShape(box=(ARENA_WIDTH + MARGIN, MARGIN)),
            ),
            self.world.CreateStaticBody(
                position=(0, ARENA_HEIGHT+MARGIN),
                shapes=polygonShape(box=(ARENA_WIDTH + MARGIN, MARGIN)),
            ),
            self.world.CreateStaticBody(
                position=(ARENA_WIDTH+MARGIN, 0),
                shapes=polygonShape(box=(MARGIN, ARENA_HEIGHT)),
            ),
            self.world.CreateStaticBody(
                position=(-MARGIN, 0),
                shapes=polygonShape(box=(MARGIN, ARENA_HEIGHT)),
            ),
            self.world.CreateStaticBody(
                position=(0.0, 0.0),
                angle = math.pi/4.0,
                shapes=polygonShape(box=(0.05, 0.05)),
            ),
            self.world.CreateStaticBody(
                position=(ARENA_WIDTH, ARENA_HEIGHT),
                angle = math.pi/4.0,
                shapes=polygonShape(box=(0.05, 0.05)),
            )
        ]
        # Randomness configuration for Monte Carlo runs
        self.randomize = randomize
        self.noise = noise or {}
        if seed is not None:
            random.seed(seed)

        # Reasonable default relative sigmas (tune as needed)
        nz = lambda k, d: self.noise.get(k, d)
        core_radius_sigma = nz('core_radius', 0.02)             # ~±5% cap via min/max mult below
        core_density_sigma = nz('core_density', 0.15)           # ~±30% cap
        core_friction_sigma = nz('core_friction', 0.25)         # ~±30% cap
        core_restitution_sigma = nz('core_restitution', 0.20)   # ~±30% cap
        core_lin_damp_sigma = nz('core_linear_damping', 0.10)   # ~±25% cap
        core_ang_damp_sigma = nz('core_angular_damping', 0.20)  # ~±50% cap

        robot_density_sigma = nz('robot_density', 0.10)         # ~±20% cap
        robot_friction_sigma = nz('robot_friction', 0.25)       # ~±30% cap
        robot_ang_damp_sigma = nz('robot_ang_damp', 0.10)       # ~±30% cap
        robot_speed_sigma = nz('robot_speed_scale', 0.05)       # ~±15% cap

        if not randomize:
            self.red_cores = [ self.world.CreateDynamicBody(position=pos) for pos in red_core_coords]
            self.green_cores = [ self.world.CreateDynamicBody(position=pos) for pos in green_core_coords]
        else:
            random_pts = GetRandomPoints(8)
            self.red_cores = [ self.world.CreateDynamicBody(position=pos) for pos in random_pts[:4]]
            self.green_cores = [ self.world.CreateDynamicBody(position=pos) for pos in random_pts[4:]]

        
        for body in self.red_cores:
            if self.randomize:
                _radius = _jitter(CORE_RADIUS, rel_sigma=core_radius_sigma, min_mult=0.95, max_mult=1.05)
                _density = _jitter(0.2, rel_sigma=core_density_sigma, min_mult=0.7, max_mult=1.3, min_val=0.05, max_val=5.0)
                _friction = _jitter(0.3, rel_sigma=core_friction_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0, max_val=1.0)
                _restitution = _jitter(0.6, rel_sigma=core_restitution_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0, max_val=1.0)
                _lin_damp = _jitter(1.1, rel_sigma=core_lin_damp_sigma, min_mult=0.8, max_mult=1.25, min_val=0.0)
                _ang_damp = _jitter(0.5, rel_sigma=core_ang_damp_sigma, min_mult=0.5, max_mult=1.5, min_val=0.0)
            else:
                _radius, _density, _friction, _restitution = CORE_RADIUS, 0.2, 0.3, 0.6
                _lin_damp, _ang_damp = 1.1, 0.5

            body.CreateCircleFixture(radius=_radius,
                density=_density, friction=_friction, restitution=_restitution,
                userData = RED_CORE_COLOR)
            body.linearDamping = _lin_damp
            body.angularDamping = _ang_damp


        
        for body in self.green_cores:
            if self.randomize:
                _radius = _jitter(CORE_RADIUS, rel_sigma=core_radius_sigma, min_mult=0.95, max_mult=1.05)
                _density = _jitter(0.2, rel_sigma=core_density_sigma, min_mult=0.7, max_mult=1.3, min_val=0.05, max_val=5.0)
                _friction = _jitter(0.3, rel_sigma=core_friction_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0, max_val=1.0)
                _restitution = _jitter(0.6, rel_sigma=core_restitution_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0, max_val=1.0)
                _lin_damp = _jitter(1.1, rel_sigma=core_lin_damp_sigma, min_mult=0.8, max_mult=1.25, min_val=0.0)
                _ang_damp = _jitter(0.5, rel_sigma=core_ang_damp_sigma, min_mult=0.5, max_mult=1.5, min_val=0.0)
            else:
                _radius, _density, _friction, _restitution = CORE_RADIUS, 0.2, 0.3, 0.6
                _lin_damp, _ang_damp = 1.1, 0.5

            body.CreateCircleFixture(radius=_radius,
                density=_density, friction=_friction, restitution=_restitution,
                userData = GREEN_CORE_COLOR)
            body.linearDamping = _lin_damp
            body.angularDamping = _ang_damp


        
        self.robots = [self.world.CreateDynamicBody(position=coord[0], angle=coord[1]) for coord in robo_coords]
        for body in self.robots:
            if self.randomize:
                _rob_density = _jitter(3.0, rel_sigma=robot_density_sigma, min_mult=0.8, max_mult=1.2, min_val=0.1, max_val=20.0)
                _rob_friction = _jitter(0.3, rel_sigma=robot_friction_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0, max_val=1.0)
                _rob_ang_damp = _jitter(100.0, rel_sigma=robot_ang_damp_sigma, min_mult=0.7, max_mult=1.3, min_val=0.0)
                body.speed_scale = _jitter(1.0, rel_sigma=robot_speed_sigma, min_mult=0.85, max_mult=1.15, min_val=0.5, max_val=1.5)
            else:
                _rob_density, _rob_friction, _rob_ang_damp = 3.0, 0.3, 100.0
                body.speed_scale = 1.0
            body.CreatePolygonFixture(box=(ROBO_LEN/2.0, ROBO_WIDTH/2.0), density=_rob_density, friction=_rob_friction)
            body.angularDamping = _rob_ang_damp


        self.robots[0].fixtures[0].userData = TEAM1_COLOR
        self.robots[1].fixtures[0].userData = TEAM1_COLOR
        self.robots[2].fixtures[0].userData = TEAM2_COLOR
        self.robots[3].fixtures[0].userData = TEAM2_COLOR

        self.scores = [0,0]
        self.red_core_counts = [0,0]
        self.simulation_time = 0

    def apply_rules(self):
        in_goals = categorize(goal_state, self.red_cores)
        self.red_cores[:] = in_goals[0]
        self.scores[0] -= len(in_goals[1])
        self.scores[1] -= len(in_goals[2])
        self.red_core_counts[0] += len(in_goals[1])
        self.red_core_counts[1] += len(in_goals[2])
        for x in in_goals[1]:
            self.world.DestroyBody(x)
        for x in in_goals[2]:
            self.world.DestroyBody(x)

        in_goals = categorize(goal_state, self.green_cores)
        self.green_cores[:] = in_goals[0]
        self.scores[0] += len(in_goals[1])
        self.scores[1] += len(in_goals[2])   
        for x in in_goals[1]:
            self.world.DestroyBody(x)
        for x in in_goals[2]:
            self.world.DestroyBody(x)
    def is_game_over(self):
        return self.red_core_counts[0] >= 3 \
            or self.red_core_counts[1] >= 3 \
            or (len(self.green_cores) == 0 and len(self.red_cores) == 0) \
            or self.simulation_time >= TIME_LIMIT
    def get_winner(self):
        if not self.is_game_over():
            return 0
        if self.red_core_counts[0] >= 3:
            return 2
        if self.red_core_counts[1] >= 3:
            return 1
        if self.scores[0] > self.scores[1]:
            return 1
        if self.scores[0] < self.scores[1]:
            return 2
        return 0
    
    def steer_robots(self):
        red_coords=[bvec_to_tuple(core.position) for core in self.red_cores]
        green_coords=[bvec_to_tuple(core.position) for core in self.green_cores]
        bot_coords=[(bvec_to_tuple(bot.position), bot.angle) for bot in self.robots]
        for robot, controller in zip(self.robots, self.controllers):
            left_vel, right_vel = controller.get_controls(bot_coords, green_coords, red_coords)
            # Apply per-robot actuator variability when randomize is on
            scale = getattr(robot, "speed_scale", 1.0)
            steer(robot, left_vel * scale * MAX_VEL, right_vel * scale * MAX_VEL)

    def step_physics(self):
        velocityIterations=10
        positionIterations=10
        self.world.Step(TIME_STEP, velocityIterations, positionIterations)
        self.world.ClearForces()
        self.simulation_time += 1.0 / TARGET_FPS
    def update(self):
        if not self.is_game_over():
            self.steer_robots()
            self.step_physics()
            self.apply_rules()
