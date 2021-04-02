import math
from .vec2d import vec_sub, vec_len, vec_mul, vec_add, vec_dist, vec_projectOn

def clamp(val, minimum, maximum):
    return min(max(val, minimum), maximum)

def normalize_angle(angle):
    if angle > math.pi:
        return angle - 2.0*math.pi
    if angle < -math.pi:
        return angle + 2.0*math.pi  
    return angle

def project_on_line_segment(seg_a, seg_b, point):
    direction = vec_sub(seg_b, seg_a)
    distance = vec_len(direction)
    if distance == 0.0:
        return seg_a
    direction = vec_mul(direction, 1.0/distance)
    projected = vec_projectOn(direction, vec_sub(point, seg_a))
    projected = clamp(projected, 0.0, distance)
    closest_point = vec_add(seg_a, vec_mul(direction, projected))
    return closest_point

def distance_to_line_segment(seg_a, seg_b, point):
    return vec_dist(project_on_line_segment(seg_a, seg_b, point), point)

def smoothstep(x):
    if x <= 0.0:
        return 0.0
    if x>=1.0:
        return 1.0
    return 3.0 * x * x - 2.0 * x * x * x