"""
    Angles in Pygame
    A demonstration by Max Goldstein, Tufts University '14
    In pygame, +y is down. Python's arctangent functions expect +y to be up.
    This wreaks havoc with the unit circle if you want to find the angle between
    two points (for, say, collision detection or aiming a gun).  You can avoid the
    problem entirely by calling atan2(-y, x) and adding 2*pi to the result if it's
    negative.
    Note that math.atan2(numerator, denominator) does the division for you.
    Controls: move the mouse near the axes.
    """

import pygame, sys, os
from pygame.locals import *
from math import atan2, degrees, pi

def quit():
    pygame.quit()
    sys.exit()

pygame.init()
screenDimensions = (400, 540)
window = pygame.display.set_mode(screenDimensions)
pygame.display.set_caption('Angles in Pygame')
screen = pygame.display.get_surface()

clock = pygame.time.Clock()
FPS = 50
time_passed = 0

pos = (200, 200)
origin = (200, 200)

white = (255, 255, 255)
black = (0, 0, 0)
red   = (255, 0, 0)
blue  = (0, 0, 255)
purple= (200, 0, 200)
green = (0, 200, 0)

font = pygame.font.Font(None, 30)

arcRect = pygame.Rect(180, 180, 40, 40)

#Game loop
while True:
    time_passed = clock.tick(FPS)
    
    #Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                quit()
        elif event.type == MOUSEMOTION:
            if 0 < event.pos[0] < 400 and 0 < event.pos[1] < 400:
                pos = event.pos

    print("mouse position:", pos)
    #Angle logic
    dx = pos[0] - origin[0]
    dy = pos[1] - origin[1]
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    if degs > 180:
        degs = degs - 360
    
    screen.fill(white)
    
    #Draw coordinate axes and labels
    pygame.draw.circle(screen, black, origin, 5)
    pygame.draw.circle(screen, black, (0,0), 5)
    pygame.draw.line(screen, black, (0, 440), (440, 440), 3)
    pygame.draw.line(screen, black, (200, 15), (200, 380))
    screen.blit(font.render( "pi/2", True, black), (178, 10))
    screen.blit(font.render("3pi/2", True, black), (175, 380))
    pygame.draw.line(screen, black, (28, 200), (355, 200))
    screen.blit(font.render("pi", True, black), (8,  190))
    screen.blit(font.render("0", True, black), (360, 190))
    
    
    #Draw lines to cursor
    pygame.draw.line(screen, blue, origin, (pos[0], origin[1]), 2)
    pygame.draw.line(screen, red, (pos[0], origin[1]), (pos[0], pos[1]), 2)
    pygame.draw.line(screen, purple, origin, pos, 2)
    pygame.draw.arc(screen, green, arcRect, 0, rads, 4)
    #Note that the function expects angles in radians
    
    #Draw numeric readings
    screen.blit(font.render(str(dx)+" x", True, blue), (10, 450))
    screen.blit(font.render(str(dy)+" y", True, red), (100, 450))
    screen.blit(font.render(str((dx**2 + dy**2)**.5)+" d", True, purple), (10, 480))
    screen.blit(font.render(str(degs)+" degrees", True, green), (10, 510))
    
    pygame.display.flip()