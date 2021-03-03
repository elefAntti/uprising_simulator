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

class SimpleBot:
    def __init__(self, index):
        self._index=index
        self._goingToBase = False
    def getWaitPosition(self):
        if self._index < 2:
            return (0.4, 1.1)
        else:
            return (0.4, 1.1)
    def getControls(self, bot_coords, green_coords, red_coords):
        own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        own_base_dist = vec_dist(get_base_coords(self._index), own_coords)
        if self._goingToBase:
            own_base_dist += 0.5
        else:
            pass
        possible_red = [x for x in red_coords if vec_dist(get_base_coords(self._index), x) > own_base_dist]
        possible_green = [x for x in green_coords if vec_dist(get_base_coords(self._index), x) < own_base_dist]
        target = self.getWaitPosition()
        if len(red_coords) == 0:
            target = (target[1], target[0])
        if possible_red:
            target = min(possible_red, key = vec_distTo(own_coords))
            self._goingToBase = False
        elif possible_green:
            target = min(possible_green, key = vec_distTo(own_coords))
            self._goingToBase = True
        else:
            self._goingToBase = True
        return steer_to_target(own_coords, own_dir, target)


class SimpleBot2:
    def __init__(self, index):
        self._index=index
        self._goingToBase = False
        self.own_coords = (0.0,0.0)
    def getBaseCoords(self):
        if self._index < 2:
            return (0.0, 1.5)
        else:
            return (1.5, 0.0)
    def getWaitPosition(self):
        if self._index < 2:
            return (0.4, 1.1)
        else:
            return (0.4, 1.1)
    def getPartnerIndex(self):
        return [1,0,3,2][self._index]
    def score(self, target):
        dist_to_self = vec_dist(target, self.own_coords)
        dist_to_partner = vec_dist(target, self.partner_coords)
        return dist_to_self * 2.0 - dist_to_partner
    def getControls(self, bot_coords, green_coords, red_coords):
        self.own_coords = bot_coords[self._index][0]
        self.partner_coords = bot_coords[self.getPartnerIndex()][0]
        own_dir = bot_coords[self._index][1]
        own_base_dist = vec_dist(self.getBaseCoords(), self.own_coords)
        if self._goingToBase:
            own_base_dist += 0.5
        else:
            pass
        possible_red = [x for x in red_coords if vec_dist(self.getBaseCoords(), x) > own_base_dist]
        possible_green = [x for x in green_coords if vec_dist(self.getBaseCoords(), x) < own_base_dist]
        target = self.getWaitPosition()
        if len(red_coords) == 0:
            target = (target[1], target[0])
        if possible_red:
            target = min(possible_red, key = lambda x: self.score(x))
            self._goingToBase = False
        elif possible_green:
            target = min(possible_green, key = lambda x: self.score(x))
            self._goingToBase = True
        else:
            self._goingToBase = True
        
        return steer_to_target(self.own_coords, own_dir, target)

class SimpleBot3:
    def __init__(self, index):
        self._index=index
        self._goingToBase = False
        self.own_coords = (0.0,0.0)
    def getBaseCoords(self):
        if self._index < 2:
            return (0.0, 1.5)
        else:
            return (1.5, 0.0)
    def getAttackDir(self):
        if self._index < 2:
            return (1.0, -1.0)
        else:
            return (-1.0, 1.0)
    def getWaitPosition(self):
        if self._index < 2:
            return max((0.1, 0.8), (0.7, 1.4), key = vec_distTo(self.partner_coords))
        else:
            return max((0.8, 0.1), (1.4, 0.7), key = vec_distTo(self.partner_coords))

    def getPartnerIndex(self):
        return [1,0,3,2][self._index]
    def score(self, target):
        dist_to_self = vec_dist(target, self.own_coords)
        dist_to_partner = vec_dist(target, self.partner_coords)
        #dist_to_center_line = vec_projectOn((1.0, 1.0), vec_sub(target, self.getBaseCoords()))
        #dist_along_center_line = vec_projectOn(self.getAttackDir(), vec_sub(target, self.getBaseCoords()))
        score = dist_to_self * 2.0 - dist_to_partner
        #if bots_closer_than(target, self.other_bots, 0.4) > 1:
        #    score -= 3.0 * dist_to_neutral_corner(target) 
        return score
    def getControls(self, bot_coords, green_coords, red_coords):
        self.own_coords = bot_coords[self._index][0]
        self.partner_coords = bot_coords[self.getPartnerIndex()][0]
        self.other_bots = [x[0] for x in other_bots(bot_coords, self._index)]
        own_dir = bot_coords[self._index][1]
        own_base_dist = vec_dist(self.getBaseCoords(), self.own_coords)
        own_base_dist_green = own_base_dist
        if self._goingToBase:
            own_base_dist += 0.5
        else:
            own_base_dist_green -= 0.5
        possible_red = [x for x in red_coords if vec_dist(self.getBaseCoords(), x) > own_base_dist]
        possible_red = [x for x in possible_red if vec_dist(self.partner_coords, x) > 0.3]
        possible_green = [x for x in green_coords if vec_dist(self.getBaseCoords(), x) < own_base_dist_green]
        possible_green = [x for x in possible_green if vec_dist(self.partner_coords, x) > 0.3]
        target = self.getWaitPosition()
        if len(red_coords) == 0:
            target = (target[1], target[0])
        if possible_red:
            target = min(possible_red, key = lambda x: self.score(x))
            self._goingToBase = False
        elif possible_green:
            target = min(possible_green, key = lambda x: self.score(x))
            self._goingToBase = True
        else:
            self._goingToBase = True
        
        return steer_to_target(self.own_coords, own_dir, target)

class Goalie:
    def __init__(self, index):
        self._index=index
        self._own_base = get_base_coords(self._index)
        self._attack_dir = vec_normalize(vec_sub((0.75, 0.75), self._own_base))
        self._post = vec_add(self._own_base, vec_mul(self._attack_dir, 0.44))
    def getControls(self, bot_coords, green_coords, red_coords):
        own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        x_dir = vec_90deg(self._attack_dir)
        closest_red = min(red_coords, key=vec_distTo(self._own_base))
        t = vec_dot(vec_sub(closest_red, self._post), x_dir)
        t = clamp(t, -.3, .3)
        target = vec_add(self._post, vec_mul(x_dir, t))
        return steer_to_target2(own_coords, own_dir, target)