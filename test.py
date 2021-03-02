#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
from pygame.locals import (QUIT, KEYDOWN, KEYUP, K_ESCAPE, K_UP, K_DOWN, K_LEFT, K_RIGHT)

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, dot)

import math

from collections import defaultdict

from bots.simple import SimpleBot, SimpleBot2, SimpleBot3

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 300.0  # pixels per meter
T_SCALE = 4.0
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS / T_SCALE
ARENA_WIDTH, ARENA_HEIGHT = 1.5, 1.5
MARGIN = 0.05
SCREEN_WIDTH, SCREEN_HEIGHT = int((ARENA_WIDTH + 2.0*MARGIN)*PPM), int((ARENA_HEIGHT + 2.0*MARGIN)*PPM)

STEER_FORCE = 1 * T_SCALE * T_SCALE
MAX_VEL = 0.5 * T_SCALE

ROBO_WIDTH = 0.11
ROBO_LEN = 0.13

TEAM1_COLOR = (255, 255, 127, 255)
TEAM2_COLOR = (127, 255, 255, 255)
RED_CORE_COLOR = (255, 128, 128, 255)
GREEN_CORE_COLOR = (128, 255, 128, 255)

red_core_coords=[
    (ARENA_WIDTH/2.0, 0.3),
    (0.15, ARENA_HEIGHT/2.0),
    (ARENA_WIDTH - 0.15, ARENA_HEIGHT/2.0),
    (ARENA_WIDTH/2.0, ARENA_HEIGHT - 0.3)
]

green_core_coords=[
    (0.15, 0.15),
    (ARENA_WIDTH/2.0 - 0.15, ARENA_HEIGHT/2.0 - 0.15),
    (ARENA_WIDTH/2.0 + 0.15, ARENA_HEIGHT/2.0 + 0.15),
    (ARENA_WIDTH - 0.15, ARENA_HEIGHT - 0.15),
]

def vec_add(a,b):
    return (a[0] + b[0], a[1] + b[1])

def vec_sub(a,b):
    return (a[0] - b[0], a[1] - b[1])

def vec_mul(a, b):
    return (a[0] * b, a[1] * b)

def vec_dot(a,b):
    return a[0]*b[0]+a[1]*b[1]

def vec_len(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1])

def vec_normalize(a):
    return vec_mul(a, 1.0/vec_len(a))

def vec_move(orig, dir, amount):
    return vec_add(orig, vec_mul(vec_normalize(dir), amount))

def vec_projectOn(on, vec):
    return vec_dot(vec, vec_normalize(on))

robo_coords=[
    (vec_move(vec_move((0.40, ARENA_HEIGHT), (-1.0,-1.0), 0.1 + ROBO_WIDTH/2.0),
        (1.0,-1.0), ROBO_LEN/2.0), -math.pi*3.0/4.0),
    (vec_move(vec_move((0.0, ARENA_HEIGHT - 0.40), (1.0,1.0), 0.1 + ROBO_WIDTH/2.0),
        (1.0,-1.0), ROBO_LEN/2.0), -math.pi*3.0/4.0),
    (vec_move(vec_move((ARENA_WIDTH - 0.40, 0.0), (1.0, 1.0), 0.1 + ROBO_WIDTH/2.0),
        (-1.0,1.0), ROBO_LEN/2.0), math.pi/4.0),
    (vec_move(vec_move((ARENA_WIDTH, 0.40), (-1.0, -1.0), 0.1 + ROBO_WIDTH/2.0),
        (-1.0,1.0), ROBO_LEN/2.0), math.pi/4.0)
]
# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('RobotUprising sim')
clock = pygame.time.Clock()

# --- pybox2d world setup ---
# Create the world
world = world(gravity=(0, 0), doSleep=True)

# And a static body to hold the ground shape
walls = [
    world.CreateStaticBody(
        position=(0, -MARGIN),
        shapes=polygonShape(box=(ARENA_WIDTH + MARGIN, MARGIN)),
    ),
    world.CreateStaticBody(
        position=(0, ARENA_HEIGHT+MARGIN),
        shapes=polygonShape(box=(ARENA_WIDTH + MARGIN, MARGIN)),
    ),
    world.CreateStaticBody(
        position=(ARENA_WIDTH+MARGIN, 0),
        shapes=polygonShape(box=(MARGIN, ARENA_HEIGHT)),
    ),
    world.CreateStaticBody(
        position=(-MARGIN, 0),
        shapes=polygonShape(box=(MARGIN, ARENA_HEIGHT)),
    ),
    world.CreateStaticBody(
        position=(0.0, 0.0),
        angle = math.pi/4.0,
        shapes=polygonShape(box=(0.05, 0.05)),
    ),
    world.CreateStaticBody(
        position=(ARENA_WIDTH, ARENA_HEIGHT),
        angle = math.pi/4.0,
        shapes=polygonShape(box=(0.05, 0.05)),
    )
]

# Create a couple dynamic bodies
red_cores = [ world.CreateDynamicBody(position=pos) for pos in red_core_coords]
for body in red_cores:
    circle = body.CreateCircleFixture(radius=0.036,
        density=0.2, friction=0.3, restitution=0.6,
        userData = RED_CORE_COLOR)
    body.linearDamping = 1.1

green_cores = [ world.CreateDynamicBody(position=pos) for pos in green_core_coords]
for body in green_cores:
    circle = body.CreateCircleFixture(radius=0.036,
        density=0.2, friction=0.3, restitution=0.6,
        userData = GREEN_CORE_COLOR)
    body.linearDamping = 1.1

robots = [world.CreateDynamicBody(position=coord[0], angle=coord[1]) for coord in robo_coords]
for body in robots:
    body.CreatePolygonFixture(box=(ROBO_WIDTH/2.0, ROBO_LEN/2.0), density=3, friction=0.3)
    body.angularDamping = 100.0

robots[0].fixtures[0].userData = TEAM1_COLOR
robots[1].fixtures[0].userData = TEAM1_COLOR
robots[2].fixtures[0].userData = TEAM2_COLOR
robots[3].fixtures[0].userData = TEAM2_COLOR

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
}

# Let's play with extending the shape classes to draw for us.

def to_screen_coords(position):
    return ((position[0] + MARGIN) * PPM, SCREEN_HEIGHT - (position[1] + MARGIN) * PPM)

def my_draw_polygon(polygon, body, fixture):
    vertices = [to_screen_coords(body.transform * v) for v in polygon.vertices]
    color = fixture.userData or colors[body.type]
    pygame.draw.polygon(screen, color, vertices)
polygonShape.draw = my_draw_polygon


def my_draw_circle(circle, body, fixture):
    position = to_screen_coords(body.transform * circle.pos)
    pygame.draw.circle(screen, fixture.userData, [int(
        x) for x in position], int(circle.radius * PPM))
    # Note: Python 3.x will enforce that pygame get the integers it requests,
    #       and it will not convert from float.
circleShape.draw = my_draw_circle

def draw_polygon(vertices, color):
    vertices = [to_screen_coords(v) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)

def draw_bases():
    draw_polygon([
        (0.0, ARENA_HEIGHT - 0.4),
        (0.0, ARENA_HEIGHT - 0.41),
        (0.41, ARENA_HEIGHT),
        (0.40, ARENA_HEIGHT)],
        TEAM1_COLOR)
    draw_polygon([
        (ARENA_WIDTH - 0.4, 0.0),
        (ARENA_WIDTH - 0.41, 0.0),
        (ARENA_WIDTH, 0.41),
        (ARENA_WIDTH, 0.40)],
        TEAM2_COLOR)

def steer_point(body, point, speed):
    forward = body.GetWorldVector((0,1))
    velocity = body.GetLinearVelocityFromWorldPoint(point)
    currentSpeed = dot(velocity, forward)
    err = currentSpeed - speed
    k = -7.0
    body.ApplyForce(forward * min(STEER_FORCE, max(-STEER_FORCE, k*err)), point, True)

def steer(body, left_speed, right_speed):
    
    steer_point(body, body.GetWorldPoint((-0.05, 0)), left_speed)
    steer_point(body, body.GetWorldPoint((0.05, 0)), right_speed)

    side = body.GetWorldVector((1,0))
    transSpeed = dot(body.linearVelocity, side)
    impulse=-transSpeed*body.mass*side
    body.ApplyLinearImpulse(impulse=impulse, point=body.position, wake=False)

class Console:
    def __init__(self):
        self.vel = 0.0
        self.turn = 0.0
    def handleEvent(self, event):
        if (event.type == KEYDOWN and event.key == K_UP):
            self.vel = 1.0
        if (event.type == KEYDOWN and event.key == K_DOWN):
            self.vel = -1.0
        if event.type == KEYUP and event.key == K_UP:
            self.vel = 0.0
        if event.type == KEYUP and event.key == K_DOWN:
            self.vel = 0.0
        if (event.type == KEYDOWN and event.key == K_LEFT):
            self.turn = 1.0
        if (event.type == KEYDOWN and event.key == K_RIGHT):
            self.turn = -1.0
        if event.type == KEYUP and event.key == K_LEFT:
            self.turn = 0.0
        if event.type == KEYUP and event.key == K_RIGHT:
            self.turn = 0.0
    def getControls(self, *args):
        left_vel = (self.vel - self.turn)
        right_vel = (self.vel + self.turn)
        scale = 1.0/max(abs(left_vel), abs(right_vel), 1.0)
        left_vel *= scale
        right_vel *= scale
        return left_vel,right_vel

class NullController:
    def getControls(self, *args):
        return 0.0,0.0

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
    if det < 0.4:
        return 1
    if det > (ARENA_HEIGHT + ARENA_WIDTH - 0.4):
        return 2
    return 0


scores = [0,0]
red_core_counts = [0,0]

def apply_rules(world, red_cores, green_cores, scores, red_core_counts):
    in_goals = categorize(goal_state, red_cores)
    red_cores[:] = in_goals[0]
    scores[0] -= len(in_goals[1])
    scores[1] -= len(in_goals[2])
    red_core_counts[0] += len(in_goals[1])
    red_core_counts[1] += len(in_goals[2])
    for x in in_goals[1]:
        world.DestroyBody(x)
    for x in in_goals[2]:
        world.DestroyBody(x)
    
    if red_core_counts[0] >= 3:
        return 2
    if red_core_counts[1] >= 3:
        return 1
    
    in_goals = categorize(goal_state, green_cores)
    green_cores[:] = in_goals[0]
    scores[0] += len(in_goals[1])
    scores[1] += len(in_goals[2])   
    for x in in_goals[1]:
        world.DestroyBody(x)
    for x in in_goals[2]:
        world.DestroyBody(x)

    if len(green_cores) == 0 and len(red_cores) == 0:
        if scores[0] > scores[1]:
            return 1
        if scores[0] < scores[1]:
            return 2
    return 0

console = Console()

controllers=[ SimpleBot2(0), SimpleBot2(1), SimpleBot3(2),SimpleBot3(3)]
#controllers=[ SimpleBot(0), SimpleBot(1), NullController(),NullController()]
# --- main game loop ---
vel = 0
turn = 0
running = False

while not running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_UP):
            # The user closed the window or pressed escape
            running = True

while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False
        console.handleEvent(event)
    red_coords=[core.position for core in red_cores]
    green_coords=[core.position for core in green_cores]
    bot_coords=[(bot.position, bot.angle) for bot in robots]
    for robot, controller in zip(robots, controllers):
        left_vel, right_vel = controller.getControls(bot_coords, green_coords, red_coords)
        steer(robot, left_vel * MAX_VEL, right_vel * MAX_VEL)        
    screen.fill((0, 0, 0, 0))
    draw_bases()
    apply_rules(world, red_cores, green_cores, scores, red_core_counts)
    pygame.display.set_caption('RobotUprising score:{} - {}, reds {} - {}'\
        .format(scores[0], scores[1], red_core_counts[0], red_core_counts[1]))
    # Draw the world
    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)
    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)
    world.ClearForces()
    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
