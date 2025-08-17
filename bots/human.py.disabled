from bots import register_bot, keyboard_listeners
from pygame.locals import (KEYDOWN, KEYUP, K_UP, K_DOWN, K_LEFT, K_RIGHT)

@register_bot
class Human:
    def __init__(self, index):
        self.vel = 0.0
        self.turn = 0.0
        keyboard_listeners.append(self)
    def handle_event(self, event):
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
    def get_controls(self, *args):
        left_vel = (self.vel - self.turn)
        right_vel = (self.vel + self.turn)
        scale = 1.0/max(abs(left_vel), abs(right_vel), 1.0)
        left_vel *= scale
        right_vel *= scale
        return left_vel,right_vel