import math
from utils.vec2d import *
from utils.math_utils import *

def dist_to_neutral_corner(pos):
    return min(vec_dist((0.0, 0.0), pos), vec_dist((1.5, 1.5), pos))

def bots_closer_than(pos, bots, dist):
    return len([x for x in bots if vec_dist(x, pos) < dist])

def other_bots(bots, own_idx):
    return [x for i,x in enumerate(bots) if i != own_idx]

def steer_to_target(own_coords, own_dir, target):
    to_target = vec_sub(target, own_coords)
    turn = normalize_angle(vec_angle(to_target) - own_dir)

    if abs(turn) < 0.1:
        return 1.0, 1.0
    if turn < 0.0:
        return 1.0, -1.0
    else:
        return -1.0, 1.0

def steer_to_target2(own_coords, own_dir, target):
    to_target = vec_sub(target, own_coords)
    turn = normalize_angle(vec_angle(to_target) - own_dir)
    turn2 = normalize_angle(vec_angle(to_target) - own_dir + math.pi)
    fw = (1.0, 1.0)
    if abs(turn2) < abs(turn):
        turn = turn2
        fw = (-1.0, -1.0)
    ratio = min(abs(turn) / 0.1, 1.0)
    ratio = ratio * ratio
    turning = (-1.0, 1.0)
    if turn < 0.0:
        turning = (1.0, -1.0)
    ratio2 = min(vec_len(to_target) / 0.02, 1.0)
    return vec_mul(vec_mix(fw, turning, ratio), ratio2)

def get_base_coords(bot_index):
    if bot_index < 2:
        return (0.0, 1.5)
    else:
        return (1.5, 0.0)

def point_in_arena(point):
    return point[0] >= 0.0 and point[0] <= 1.5 \
        and point[1] >= 0.0 and point[1] <= 1.5 

def get_partner_index(index):
    return [1,0,3,2][index]

def get_opponent_index(index):
    return [2,3,0,1][index]

def pairs(elems):
    for i in range(len(elems)):
        for j in range(i):
            yield (elems[i], elems[j])