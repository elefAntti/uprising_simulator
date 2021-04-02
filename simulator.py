from collections import defaultdict
from game_data import *
import Box2D  # The main library
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, dot)
from utils.vec2d import *
import math
import random

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
    def init(self, controllers, randomize = False):
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
        if not randomize:
            self.red_cores = [ self.world.CreateDynamicBody(position=pos) for pos in red_core_coords]
            self.green_cores = [ self.world.CreateDynamicBody(position=pos) for pos in green_core_coords]
        else:
            random_pts = GetRandomPoints(8)
            self.red_cores = [ self.world.CreateDynamicBody(position=pos) for pos in random_pts[:4]]
            self.green_cores = [ self.world.CreateDynamicBody(position=pos) for pos in random_pts[4:]]

        for body in self.red_cores:
            body.CreateCircleFixture(radius=CORE_RADIUS,
                density=0.2, friction=0.3, restitution=0.6,
                userData = RED_CORE_COLOR)
            body.linearDamping = 1.1

        for body in self.green_cores:
            body.CreateCircleFixture(radius=CORE_RADIUS,
                density=0.2, friction=0.3, restitution=0.6,
                userData = GREEN_CORE_COLOR)
            body.linearDamping = 1.1

        self.robots = [self.world.CreateDynamicBody(position=coord[0], angle=coord[1]) for coord in robo_coords]
        for body in self.robots:
            body.CreatePolygonFixture(box=(ROBO_LEN/2.0, ROBO_WIDTH/2.0), density=3, friction=0.3)
            body.angularDamping = 100.0

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
        red_coords=[core.position for core in self.red_cores]
        green_coords=[core.position for core in self.green_cores]
        bot_coords=[(bot.position, bot.angle) for bot in self.robots]
        for robot, controller in zip(self.robots, self.controllers):
            left_vel, right_vel = controller.get_controls(bot_coords, green_coords, red_coords)
            steer(robot, left_vel * MAX_VEL, right_vel * MAX_VEL)
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
