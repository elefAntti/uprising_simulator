#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
from pygame.locals import (QUIT, KEYDOWN, KEYUP, K_ESCAPE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RETURN)
from game_data import *
from vec2d import *
import math
import Box2D  # The main library
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody)
from simulator import Simulator
from bots.simple import SimpleBot, SimpleBot2, SimpleBot3, Prioritiser2, Prioritiser

SCREEN_WIDTH, SCREEN_HEIGHT = int((ARENA_WIDTH + 2.0*MARGIN)*PPM), int((ARENA_HEIGHT + 2.0*MARGIN)*PPM)

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
}

pygame.font.init()
fontname = pygame.font.match_font('arial')
font = pygame.font.Font(fontname, 24)
big_font = pygame.font.Font(fontname, 48)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('RobotUprising sim')
clock = pygame.time.Clock()

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

def draw_winner(winner):
    color = (128,128,128,255)
    message = "Draw"
    if winner == 1:
        color = TEAM1_COLOR
        message = "Team 1 won"

    if winner == 2:
        color = TEAM2_COLOR
        message = "Team 2 won"

    msg_pic=big_font.render(message, False, color)
    screen.blit(msg_pic,(100, SCREEN_HEIGHT/2 - 50)) 

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

def draw_scores(scores, red_core_counts):
    message='score: {} reds: {}'.format(scores[0], red_core_counts[0])
    msg_pic=font.render(message, False, TEAM1_COLOR)
    screen.blit(msg_pic,(150,20))
    message='score: {} reds: {}'.format(scores[1], red_core_counts[1])
    msg_pic=font.render(message, False, TEAM2_COLOR)
    screen.blit(msg_pic,(150, SCREEN_HEIGHT - 50))

console = Console()

#controllers=[Prioritiser(0), Prioritiser(1), Prioritiser2(2), Prioritiser2(3)]
controllers=[SimpleBot2(0), console, Prioritiser2(2), Prioritiser2(3)]
#controllers=[ NullController(),  NullController(),console,NullController()]

# --- main game loop ---
running = True
finished = False
winner = 0
simulator = Simulator()
simulator.init(controllers)

while running:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
        if event.type == KEYDOWN and event.key == K_RETURN:
            simulator.init(controllers)
        console.handleEvent(event)
    simulator.update()
    finished = simulator.is_game_over()
    winner = simulator.get_winner()
    screen.fill((0, 0, 0, 0))
    draw_bases()
    draw_scores(simulator.scores, simulator.red_core_counts)
    for body in simulator.world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)
    if finished:
        draw_winner(winner)
    pygame.display.flip()
    clock.tick(TARGET_FPS)
pygame.quit()
