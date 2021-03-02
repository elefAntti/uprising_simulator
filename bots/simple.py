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

def dist_to_neutral_corner(pos):
    return min(vec_dist((0.0, 0.0), pos), vec_dist((1.5, 1.5), pos))

def bots_closer_than(pos, bots, dist):
    return len([x for x in bots if vec_dist(x, pos) < dist])

def other_bots(bots, own_idx):
    return [x for i,x in enumerate(bots) if i != own_idx]

class SimpleBot:
    def __init__(self, index):
        self._index=index
        self._goingToBase = False
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
    def getControls(self, bot_coords, green_coords, red_coords):
        own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        own_base_dist = vec_dist(self.getBaseCoords(), own_coords)
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
            target = min(possible_red, key = vec_distTo(own_coords))
            self._goingToBase = False
        elif possible_green:
            target = min(possible_green, key = vec_distTo(own_coords))
            self._goingToBase = True
        else:
            self._goingToBase = True
        to_target = vec_sub(target, own_coords)
        turn = vec_angle(to_target) - own_dir - math.pi/2.0

        if abs(turn) < 0.1:
            return 1.0, 1.0
        if turn < 0.0:
            return 1.0, -1.0
        else:
            return -1.0, 1.0


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
        
        to_target = vec_sub(target, self.own_coords)
        turn = vec_angle(to_target) - own_dir - math.pi/2.0

        if abs(turn) < 0.1:
            return 1.0, 1.0
        if turn < 0.0:
            return 1.0, -1.0
        else:
            return -1.0, 1.0

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
            return (0.4, 1.1)
        else:
            return (0.4, 1.1)
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
        possible_green = [x for x in green_coords if vec_dist(self.getBaseCoords(), x) < own_base_dist_green]
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
        
        to_target = vec_sub(target, self.own_coords)
        turn = vec_angle(to_target) - own_dir - math.pi/2.0

        if abs(turn) < 0.1:
            return 1.0, 1.0
        if turn < 0.0:
            return 1.0, -1.0
        else:
            return -1.0, 1.0