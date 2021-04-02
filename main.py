#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate the game played at RobotUprising hackathon

Two teams of robots (differential drive with tracks) compete in an arena,
their position and the position of energy cores (balls) are tracked from top using machine vision.

There are two bots per team, the objective is to push the red balls to opponents corner and
green balls to your own corner. If a player gets three green balls in their corner, they lose.
Otherwise the game ends as all the balls have been scored or after a time limit (not implemented yet)

In the bots.simple module there are couple of simple test programs to play against.
You can select "Human" as the player to control that bot with arrow keys.

Other controls:
    ESC Quit
    SPACE pause
    ENTER reset

Author: Antti Valli (https://github.com/elefAntti)
Released under BSD 2-clause licence

Copyright (c) 2021, Antti Valli
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE."""


import pygame
from pygame.locals import (QUIT, KEYDOWN, KEYUP, K_ESCAPE, K_r, K_RETURN, K_SPACE)
from game_data import *
from utils.vec2d import *
import math
import sys
import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody)
from simulator import Simulator
import bots.simple
import bots.human
import argparse
from bots import bot_type, keyboard_listeners

parser = argparse.ArgumentParser(prog='main', \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--team1', type=str, default="Team 1", help="Team 1 name")
parser.add_argument('--team2', type=str, default="Team 2", help="Team 2 name")
parser.add_argument('bots', nargs='*', default=[], help='''\
     0, 2 or 4 bot names.
     0: Play a default match
     2: Play the selected bots against each other (2 in each team)
     4: Play the selected bots against each other
     ''')

args = parser.parse_args()


team1_name = args.team1
team2_name = args.team2
 
for name in args.bots:
    if name not in bots.bot_type:
        print("'{}' isn't a registered bot class".format(name))
        parser.print_help()
        sys.exit(1)

if len(args.bots) == 0:
    player_names = ["PotentialWinner", "PotentialWinner", "SimpleBot2", "SimpleBot2"]
elif len(args.bots) == 2:
    player_names = [args.bots[0], args.bots[0], args.bots[1], args.bots[1]]
elif len(args.bots) == 4:
    player_names = args.bots
else:
    parser.print_help()
    sys.exit(1)

SCREEN_WIDTH = int((ARENA_WIDTH + 2.0*MARGIN)*PPM)
SCREEN_HEIGHT = int((ARENA_HEIGHT + 2.0*MARGIN)*PPM)

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
}

pygame.font.init()
fontname = pygame.font.match_font('arial')
font = pygame.font.Font(fontname, 24)
big_font = pygame.font.Font(fontname, 48)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Uprising simulator')
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
        message = team1_name + " won"

    if winner == 2:
        color = TEAM2_COLOR
        message = team2_name + " won"

    msg_pic=big_font.render(message, False, color)
    screen.blit(msg_pic,(100, SCREEN_HEIGHT/2 - 50)) 

def draw_info(message):
    color = (128,128,128,255)
    msg_pic=big_font.render(message, False, color)
    screen.blit(msg_pic,(100, SCREEN_HEIGHT/2 - 50)) 

def draw_scores(scores, red_core_counts):
    message='score: {} reds: {}'.format(scores[0], red_core_counts[0])
    msg_pic=font.render(message, False, TEAM1_COLOR)
    screen.blit(msg_pic,(150,20))
    message='score: {} reds: {}'.format(scores[1], red_core_counts[1])
    msg_pic=font.render(message, False, TEAM2_COLOR)
    screen.blit(msg_pic,(150, SCREEN_HEIGHT - 50))

def create_controllers():
    keyboard_listeners.clear()
    return [bot_type[player_names[i]](i) for i in range(4)]

controllers=create_controllers()

running = True
finished = False
paused = True
random = False
winner = 0
simulator = Simulator()
simulator.init(controllers, random)

while running:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
        if event.type == KEYDOWN and event.key == K_r:
            random = not random
        if event.type == KEYDOWN and (event.key == K_RETURN or event.key == K_r ):
            controllers=create_controllers()
            simulator.init(controllers, random)
        if event.type == KEYDOWN and event.key == K_SPACE:
            paused = not paused
        for listener in keyboard_listeners:
            listener.handle_event(event)
    if not paused:
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
    elif paused: 
        draw_info("Pause")
    pygame.display.flip()
    clock.tick(TARGET_FPS)
pygame.quit()
