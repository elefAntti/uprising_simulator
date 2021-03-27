import math
from collections import namedtuple
from bots.utility_functions import *
from bots import register_bot

@register_bot
class NullController:
    def get_controls(self, *args):
        return 0.0,0.0

@register_bot
class SimpleBot:
    def __init__(self, index):
        self._index=index
        self._goingToBase = False
    def getWaitPosition(self):
        if self._index < 2:
            return (0.4, 1.1)
        else:
            return (0.4, 1.1)
    def get_controls(self, bot_coords, green_coords, red_coords):
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

@register_bot
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
    def get_controls(self, bot_coords, green_coords, red_coords):
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

@register_bot
class SimpleBot3:
    def __init__(self, index, new_steer = False):
        self._index=index
        self._goingToBase = False
        self._new_steer = new_steer
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
    def get_controls(self, bot_coords, green_coords, red_coords):
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
        if self._new_steer:
            return steer_to_target2(self.own_coords, own_dir, target)
        else:
            return steer_to_target(self.own_coords, own_dir, target)

@register_bot
class Goalie:
    def __init__(self, index):
        self._index=index
        self._own_base = get_base_coords(self._index)
        self._attack_dir = vec_normalize(vec_sub((0.75, 0.75), self._own_base))
        self._post = vec_add(self._own_base, vec_mul(self._attack_dir, 0.44))
    def get_controls(self, bot_coords, green_coords, red_coords):
        own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        x_dir = vec_90deg(self._attack_dir)
        closest_red = min(red_coords, key=vec_distTo(self._own_base))
        t = vec_dot(vec_sub(closest_red, self._post), x_dir)
        t = clamp(t, -.3, .3)
        target = vec_add(self._post, vec_mul(x_dir, t))
        return steer_to_target2(own_coords, own_dir, target)

KIND_GREEN = 0
KIND_RED = 1
KIND_POINT = 2
KIND_KICK_RED = 2
KIND_HOME = 3
KIND_KICK_AWAY =4
KIND_KICK_GREEN = 5
target_type = namedtuple("Target", ["kind", "coords"])

@register_bot
class Prioritiser:
    def __init__(self, index):
        self._index=index
        self._own_coords = (0.0, 0.0)
        self._own_base = get_base_coords(self._index)
        self._opponent_base = get_base_coords(get_opponent_index(self._index))
        self._attack_dir = vec_normalize(vec_sub((0.75, 0.75), self._own_base))
        self._post = vec_add(self._own_base, vec_mul(self._attack_dir, 0.44))
    def evaluate(self, target):
        score = 0.0
        if target.kind == KIND_RED:
            dot = vec_dot(vec_normalize(vec_sub(target.coords, self._opponent_base)), \
                vec_normalize(vec_sub(self._own_coords, self._opponent_base)))
            if vec_dist(self._opponent_base, self._own_coords) < \
                vec_dist(self._opponent_base, target.coords):
                score -= 10.0 
            elif dot > 0.98: #Angle to opponent goal is good
                score += 100.0
            else:
                score += 3.0
            dist = min(distance_to_line_segment(target.coords, self._opponent_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to opponents goal is not blocked
                score += 5.0
        if target.kind == KIND_GREEN:
            dot = vec_dot(vec_normalize(vec_sub(target.coords, self._own_base)), \
                vec_normalize(vec_sub(self._own_coords, self._own_base)))
            if vec_dist(self._own_base, self._own_coords) < \
                vec_dist(self._own_base, target.coords):
                score -= 3.0 
            elif dot > 0.97: #Angle to opponent goal is good
                score += 3.0
            else:
                score += 1.0
            dist = min(distance_to_line_segment(target.coords, self._own_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to opponents goal is not blocked
                score += 5.0
        if target.kind == KIND_KICK_RED:
            dist = min(distance_to_line_segment(target.coords, self._opponent_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to opponents goal is not blocked
                score += 5.0
            score += 6.0  
        if target.kind == KIND_KICK_AWAY:
            score += 4.0
        score -= vec_dist(self._own_coords, target.coords)
        score += vec_dist(self._partner_coords, target.coords)
        return score
    def get_controls(self, bot_coords, green_coords, red_coords):
        self._own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        self._bots = [bot[0] for bot in bot_coords]
        self._partner_coords = bot_coords[get_partner_index(self._index)][0]
        self._red_coords = red_coords
        targets = [target_type(KIND_RED, coords) for coords in red_coords]
        targets += [target_type(KIND_GREEN, coords) for coords in green_coords]
        for coord in red_coords:
            kick_dir = vec_normalize(vec_sub(coord, self._opponent_base))
            kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
            if not point_in_arena(kick_pos):
                kick_dir = vec_normalize(vec_sub(self._own_base, coord))
                kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
                targets.append(target_type(KIND_KICK_AWAY, kick_pos))
            else:
                targets.append(target_type(KIND_KICK_RED, kick_pos))
        targets.append(target_type(KIND_HOME, self._post))
        target = max(targets, key=lambda x: self.evaluate(x)).coords
        return steer_to_target2(self._own_coords, own_dir, target)

@register_bot
class Prioritiser2:
    def __init__(self, index):
        self._index=index
        self._own_coords = (0.0, 0.0)
        self._target = (0.0, 0.0)
        self._own_base = get_base_coords(self._index)
        self._opponent_base = get_base_coords(get_opponent_index(self._index))
        self._attack_dir = vec_normalize(vec_sub((0.75, 0.75), self._own_base))
        self._post = vec_add(self._own_base, vec_mul(self._attack_dir, 0.44))
    def evaluate(self, target):
        score = 0.0
        if target.kind == KIND_RED:
            dot = vec_dot(vec_normalize(vec_sub(target.coords, self._opponent_base)), \
                vec_normalize(vec_sub(self._own_coords, self._opponent_base)))
            if vec_dist(self._opponent_base, self._own_coords) < \
                vec_dist(self._opponent_base, target.coords):
                score -= 10.0 
            elif dot > 0.98: #Angle to opponent goal is good
                score += 100.0
            else:
                score += 3.0
            dist = min(distance_to_line_segment(target.coords, self._opponent_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to opponents goal is not blocked
                score += 5.0
        if target.kind == KIND_GREEN:
            dot = vec_dot(vec_normalize(vec_sub(target.coords, self._own_base)), \
                vec_normalize(vec_sub(self._own_coords, self._own_base)))
            if vec_dist(self._own_base, self._own_coords) < \
                vec_dist(self._own_base, target.coords):
                score -= 3.0 
            elif dot > 0.97: #Angle to opponent goal is good
                score += 3.0
            else:
                score += 1.0
            dist = min(distance_to_line_segment(target.coords, self._own_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to own goal is not blocked
                score += 5.0
        if target.kind == KIND_KICK_RED:
            dist = min(distance_to_line_segment(target.coords, self._opponent_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to opponents goal is not blocked
                score += 5.0
            score += 6.0  
        if target.kind == KIND_KICK_GREEN:
            dist = min(distance_to_line_segment(target.coords, self._own_base, bot) for bot in self._bots)
            if dist > 0.2: # the line to own goal is not blocked
                score += 3.0
            score += 2.0 
        if target.kind == KIND_KICK_AWAY:
            score += 2.0
        score -= vec_dist(self._own_coords, target.coords)
        score += vec_dist(self._partner_coords, target.coords)
        if vec_dist(self._partner_coords, target.coords) < 0.2:
            score -= 3.0
        if vec_dist(self._target, target.coords) < 0.1: #hysteresis
            score += 1.0
        return score
    def get_controls(self, bot_coords, green_coords, red_coords):
        self._own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        self._bots = [bot[0] for bot in bot_coords]
        self._partner_coords = bot_coords[get_partner_index(self._index)][0]
        self._red_coords = red_coords
        targets = [target_type(KIND_RED, coords) for coords in red_coords]
        targets += [target_type(KIND_GREEN, coords) for coords in green_coords]
        for coord in red_coords:
            kick_dir = vec_normalize(vec_sub(coord, self._opponent_base))
            kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
            if not point_in_arena(kick_pos):
                kick_dir = vec_normalize(vec_sub(self._own_base, coord))
                kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
                targets.append(target_type(KIND_KICK_AWAY, kick_pos))
            else:
                targets.append(target_type(KIND_KICK_RED, kick_pos))
        for coord in green_coords:
            kick_dir = vec_normalize(vec_sub(coord, self._own_base))
            kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
            if not point_in_arena(kick_pos):
                kick_dir = vec_normalize(vec_sub(self._own_base, coord))
                kick_pos = vec_sub(coord, vec_mul(kick_dir, -0.3))
                #targets.append(target_type(KIND_KICK_AWAY, kick_pos))
            else:
                targets.append(target_type(KIND_KICK_GREEN, kick_pos))
       
        targets.append(target_type(KIND_HOME, self._post))
        self._target = max(targets, key=lambda x: self.evaluate(x)).coords
        return steer_to_target2(self._own_coords, own_dir, self._target)


class PotentialWinnerBase:
    """
    Calculates a potential field and steers based on its gradient
    """
    def __init__(self, index, param):
        self._index = index
        self._param = param
    def potential_at(self, coords):
        opponent_base = get_base_coords(get_opponent_index(self._index))
        own_base = get_base_coords(self._index)
        potential = 0.0
        potential -= 0.2 / vec_dist(coords, self._partner_coords)
        potential -= 1.0 / math.pow(coords[0] / 0.1, 4.0)
        potential -= 1.0 / math.pow(coords[1] / 0.1, 4.0)
        potential -= 1.0 / math.pow((1.5 - coords[0]) / 0.1, 4.0)
        potential -= 1.0 / math.pow((1.5 - coords[1]) / 0.1, 4.0)
        for r in self._red_coords:
            dot = vec_dot( vec_normalize(vec_sub(coords, opponent_base)),\
                           vec_normalize(vec_sub(coords, r))) + 1.0
            potential += dot / vec_dist(coords, r)
        for r in self._green_coords:
            dot = vec_dot(vec_normalize(vec_sub(coords, own_base)),\
                           vec_normalize(vec_sub(coords, r))) + 1.0
            potential += 0.2 * dot / vec_dist(coords, r)
        return potential
    def get_controls(self, bot_coords, green_coords, red_coords):
        self._partner_coords = bot_coords[get_partner_index(self._index)][0]
        self._red_coords = red_coords
        self._green_coords = green_coords
        self._other_bots = [bot_coords[get_opponent_index(self._index)][0],
            bot_coords[get_opponent_index(get_partner_index(self._index))][0]]
        own_coords = bot_coords[self._index][0]
        own_dir = bot_coords[self._index][1]
        forward=vec_unitInDir(own_dir)
        left=vec_90deg(forward)

        sample_point = vec_add(own_coords, vec_mul(forward, 0.05))
        d_long = self.potential_at(vec_move(sample_point, forward, 0.005))
        d_long -= self.potential_at(vec_move(sample_point, forward, -0.005))

        d_side = self.potential_at(vec_move(sample_point, left, 0.005))
        d_side -= self.potential_at(vec_move(sample_point, left, -0.005))

        left_track = -d_side + d_long
        right_track = d_side + d_long

        factor = vec_infnorm((left_track, right_track))
        return left_track/factor, right_track/factor


@register_bot
class PotentialWinner(PotentialWinnerBase):
    def __init__(self, index):
        PotentialWinnerBase.__init__(self, index, 0.1)

@register_bot
class PotentialWinner2(PotentialWinnerBase):
    def __init__(self, index):
        PotentialWinnerBase.__init__(self, index, 0.07)