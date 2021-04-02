from utils.vec2d import *
import math

PPM = 300.0  # pixels per meter
T_SCALE = 4.0 # Required hack so balls don't stick to walls 
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS / T_SCALE
ARENA_WIDTH, ARENA_HEIGHT = 1.5, 1.5
MARGIN = 0.05

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

CORE_RADIUS = 0.036

STEER_FORCE = 1 * T_SCALE * T_SCALE
MAX_VEL = 0.5 * T_SCALE

ROBO_WIDTH = 0.11
ROBO_LEN = 0.13

TIME_LIMIT = 30.0

TEAM1_COLOR = (255, 255, 127, 255)
TEAM2_COLOR = (127, 255, 255, 255)
RED_CORE_COLOR = (255, 128, 128, 255)
GREEN_CORE_COLOR = (128, 255, 128, 255)

TEAM1_BASE=(0.0, ARENA_HEIGHT)
TEAM2_BASE=(ARENA_WIDTH, 0.0)

#The distance of the robot from the edge of arena along the base
ROBO_EDGE_DIST = 0.1
BASE_SIZE = 0.4
_ROBO_CORNER = vec_move(vec_move(vec_add(TEAM1_BASE, (BASE_SIZE, 0.0)),
            (-1.0, -1.0), ROBO_EDGE_DIST), (1.0, -1.0), ROBO_LEN)
#Safety area for core spawning so they dont spawn inside robots or bases
BASE_SAFETY_AREA = vec_dist(TEAM1_BASE, _ROBO_CORNER) + CORE_RADIUS

robo_coords=[
    (vec_move(vec_move((0.40, ARENA_HEIGHT), (-1.0,-1.0), ROBO_EDGE_DIST + ROBO_WIDTH/2.0),
        (1.0,-1.0), ROBO_LEN/2.0), -math.pi/4.0),
    (vec_move(vec_move((0.0, ARENA_HEIGHT - 0.40), (1.0,1.0), ROBO_EDGE_DIST + ROBO_WIDTH/2.0),
        (1.0,-1.0), ROBO_LEN/2.0), -math.pi/4.0),
    (vec_move(vec_move((ARENA_WIDTH - 0.40, 0.0), (1.0, 1.0), ROBO_EDGE_DIST + ROBO_WIDTH/2.0),
        (-1.0,1.0), ROBO_LEN/2.0), math.pi*3.0/4.0),
    (vec_move(vec_move((ARENA_WIDTH, 0.40), (-1.0, -1.0), ROBO_EDGE_DIST + ROBO_WIDTH/2.0),
        (-1.0,1.0), ROBO_LEN/2.0), math.pi*3.0/4.0)
]