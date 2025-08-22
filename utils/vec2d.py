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

def vec_norm(a):
    return vec_len(a)

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

def vec_infnorm(a):
    return max(abs(a[0]), abs(a[1]))

def vec_average(a,b):
    return vec_mul(vec_add(a, b), 0.5)
