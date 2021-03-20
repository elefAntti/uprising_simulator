import math

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

def vec_cross(a,b):
    return a[0] * b[1] - a[1] * b[0]

def vec_unitInDir(angle):
    return (math.cos(angle), math.sin(angle))

def vec_angle(a):
    return math.atan2(a[1], a[0])

def vec_dist(a, b):
    return vec_len(vec_sub(a, b))

def vec_distTo(a):
    return lambda b: vec_dist(a,b)

def vec_90deg(a):
    return -a[1], a[0]

def vec_mix(a,b, ratio):
    return vec_add(vec_mul(a, (1.0 - ratio)), vec_mul(b, ratio))

def dist_to_neutral_corner(pos):
    return min(vec_dist((0.0, 0.0), pos), vec_dist((1.5, 1.5), pos))

def bots_closer_than(pos, bots, dist):
    return len([x for x in bots if vec_dist(x, pos) < dist])

def other_bots(bots, own_idx):
    return [x for i,x in enumerate(bots) if i != own_idx]

def clamp(val, minimum, maximum):
    return min(max(val, minimum), maximum)

def normalize_angle(angle):
    if angle > math.pi:
        return angle - 2.0*math.pi
    if angle < -math.pi:
        return angle + 2.0*math.pi  
    return angle

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

def distance_to_line_segment(seg_a, seg_b, point):
    direction = vec_sub(seg_b, seg_a)
    distance = vec_len(direction)
    direction = vec_mul(direction, 1.0/distance)
    projected = vec_projectOn(direction, vec_sub(point, seg_a))
    projected = clamp(projected, 0.0, distance)
    closest_point = vec_add(seg_a, vec_mul(direction, projected))
    return vec_dist(closest_point, point)