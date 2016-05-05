'''carmunk:
    1. GameState: initializes the board for each "game". a "game" can last 1M frames.
    2. frame_step: 
        a. moves the objects on the board
            i. moves self every frame according to prediction based on prior iteration
            ii. moves large, slow objects every 100 frames, adjusting speed, direction randomly by +/- 2 radians
            iii. moves cat/dog every 5 frames, adjusting speed, direction randomply by +/- 1 degree
        b. notes position and attitude of self
            i. sends sonar signal and measures object distances
            ii. determines reward based on those readings i.e., crashed or not
            iii. passes reward and new sensor (3) readings back
        
        '''
import random
import math
import numpy as np
import pygame
from pygame.color import THECOLORS
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw
import time
from six.moves import cPickle as pickle
from scipy import ndimage
from PIL import Image
from math import atan2, degrees, pi, sqrt

# ***** initialize variables *****

# operating modes drive neural net training
HUNT = 1
TURN = 2
SPEED = 3
ACQUIRE = 4
cur_mode = TURN

# PyGame init
width = 1000
height = 700
pygame.init()
clock = pygame.time.Clock()

# display surface
"""note: pygame "movements" are made by updating the "screen" (the bottom graphic level) and image "surfaces" (layers) above that level. you then bind the "surfaces" to the screen by "blit"-ing. finally, you "flip" the combinded package to print it to the output screen. so, you'll see screen here. you'll also see a variety of surfaces (sometimes called "grids") to accomplish those layers"""
    
screen = pygame.display.set_mode((width, height))
BACK_COLOR = "black"
WALL_COLOR = "red"
CAR_COLOR = "green"
SMILE_COLOR = "blue"
OBSTACLE_COLOR = "purple"
CAT_COLOR = "orange"
CAR_BODY_DIAM = 12
SONAR_ARM_LEN = 20
show_sensors = False # Showing sensors and redrawing slows things down.
draw_screen = False

# turn model settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings

# speed model settings
SPEED_NUM_SENSOR = 7 # front five, rear two sonar
SPEEDS = [0,30,50,70]

# acquire model settngs
target_grid = pygame.Surface((width, height), pygame.SRCALPHA, 32)
target_grid.convert_alpha()
ACQUIRE_PIXEL_COLOR = "green"
ACQUIRED_PIXEL_COLOR = "yellow"
ACQUIRE_PIXEL_SIZE = 2
ACQUIRE_PIXEL_SEPARATION = 25
ACQUIRE_MARGIN = 50
TARGET_RADIUS = 2
path_grid = pygame.Surface((width, height))
PATH_COLOR = "grey"

# hunt model settings
HUNT_NUM_SENSOR = 7
STATS_BUFFER = 2000

# future
DOWN_SAMPLE = 10

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# ***** instantiate the game *****

class GameState:
    def __init__(self):
        # crash state
        self.crashed = False

        # initialize space
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # create self a.k.a. "car"
        self.create_car(width/2, height/2, 0.5)
        self.cur_x, self.cur_y = self.car_body.position

        # initialize counters
        self.num_steps = 0
        self.num_off_scrn = 0

        # create walls
        static = [pymunk.Segment(self.space.static_body,(0, 1), (0, height), 1),
                  pymunk.Segment(self.space.static_body,(1, height), (width, height), 1),
                  pymunk.Segment(self.space.static_body,(width-1, height), (width-1, 1), 1),
                  pymunk.Segment(self.space.static_body,(1, 1), (width, 1), 1)]
        
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS[WALL_COLOR] # 'red'
        self.space.add(static)
        
        if cur_mode in [TURN, SPEED, HUNT]:

            # create slow, randomly moving, larger obstacles
            self.obstacles = []
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                   random.randint(70, height-100),50)) # was 100
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                   random.randint(70, height-70),50)) # was 100
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                   random.randint(70, height-70),63)) # was 125
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                    random.randint(70, height-70),63)) # was 125
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                    random.randint(70, height-70),30)) # was 35
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                    random.randint(70, height-70),30)) # was 35
        
            # create faster, randomly moving, smaller obstacles a.k.a. "cats"
            self.cats = []
            self.cats.append(self.create_cat(width-950,height-100))
            self.cats.append(self.create_cat(width-50,height-600))
            self.cats.append(self.create_cat(width-50,height-100))
            self.cats.append(self.create_cat(width-50,height-600))

        if cur_mode in [ACQUIRE, HUNT]:
            # set up seach grid and feed first target
            self.target_pixels = []
            self.current_target = (0,0)
            self.acquired_pixels = []
            self.generate_targets(True)
            self.last_x = width/2 + 1
            self.last_y = height/2 +1
            self.assign_next_target((self.last_x, self.last_y), True)
            self.target_acquired = False
            self.last_target_dist = 350
            self.last_obstacle_dist = 10
            self.target_deltas = [0.4,0.5,0.6]
            #self.obstacle_deltas = [1,2,3]
            self.obstacle_dists = [12,10]
        """ note: while python assumes (0,0) is in the lower left of the screen, pygame assumes (0,0) in the upper left. therefore, y+ moves DOWN the y axis. here is example code that illustrates how to handle angles in that environment: https://github.com/mgold/Python-snippets/blob/master/pygame_angles.py. in this implementation, I have flipped the screen so that Y+ moves UP the screen."""

    # ***** primary logic controller for game play *****

    def frame_step(self, cur_mode, turn_action, speed_action, cur_speed, car_distance):
        
        # plot move based on current (active) model prediction
        if cur_mode in [TURN, SPEED, ACQUIRE, HUNT]:
            # action == 0 is continue current trajectory
            if turn_action == 1:  # slight right adjust to current trajectory
                self.car_body.angle -= .2
            elif turn_action == 2:  # hard right
                self.car_body.angle -= .4
            elif turn_action == 3:  # slight left
                self.car_body.angle += .2
            elif turn_action == 4:  # hard left
                self.car_body.angle += .4
        
        if cur_mode in [SPEED, HUNT]: # setting speed valued directly see SPEEDS
            # action == 0 is no speed change
            if speed_action == 1: # stop
                cur_speed = SPEEDS[0]
            elif speed_action == 2: # 30
                cur_speed = SPEEDS[1]
            elif speed_action == 3: # 50
                cur_speed = SPEEDS[2]
            elif speed_action == 4: # 70
                cur_speed = SPEEDS[3]
        
        # effect move by applying speed and direction as vector on self
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = cur_speed * driving_direction
        
        if cur_mode in [TURN, SPEED, HUNT]:
            # move slow obstacles
            if self.num_steps % 20 == 0: # 20x slower than self
                self.move_obstacles()

            # move fast obstacles
            if self.num_steps % 40 == 0: # 40 x more stable than self
                self.move_cats()
       
       # Update the screen and surfaces
        screen.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        if cur_mode in [ACQUIRE, HUNT]:
            
            # draw the path self has taken on the acquire grid
            pygame.draw.lines(path_grid, pygame.color.THECOLORS[PATH_COLOR], True,
                              ((self.last_x, height-self.last_y),
                               (self.cur_x, height-self.cur_y)), 1)
            
            # overlay the path, target surfaces on the screen
            screen.blit(path_grid, (0,0))
            screen.blit(target_grid, (0,0))
        
        draw(screen, self.space)
        self.space.step(1./10) # one pixel for every 10 SPEED
        if draw_screen:
            pygame.display.flip()

        self.last_x = self.cur_x; self.last_y = self.cur_y
        self.cur_x, self.cur_y = self.car_body.position

        # get readings from the various sensors
        sonar_dist_readings, sonar_color_readings = \
            self.get_sonar_dist_color_readings(self.cur_x, self.cur_y, self.car_body.angle)
        turn_readings = sonar_dist_readings[:TURN_NUM_SENSOR]
        turn_readings = turn_readings + sonar_color_readings[:TURN_NUM_SENSOR]
        
        speed_readings = sonar_dist_readings[:SPEED_NUM_SENSOR]
        speed_readings.append(turn_action)
        speed_readings.append(cur_speed)

        if cur_mode in [ACQUIRE, HUNT]:
            
            # 1. calculate distance and angle to active target(s)
            # a. euclidean distance traveled
            dx = self.current_target[0] - self.cur_x
            dy = self.current_target[1] - self.cur_y
            
            target_dist = ((dx**2 + dy**2)**0.5)
            
            # b. calculate target angle
                # i. relative to car
            rads = atan2(dy,dx)
            rads %= 2*pi
            target_angle_degs = degrees(rads)
            
            if target_angle_degs > 180:
                target_angle_degs = target_angle_degs - 360
            
                # ii. relative to car's current direction
            rads = self.car_body.angle
            rads %= 2*pi
            car_angle_degs = degrees(rads)
            
            if car_angle_degs > 360:
                car_angle_degs = car_angle_degs - 360
            
            # "heading" accounts for angle from car and of car netting degrees car must turn
            heading_to_target = target_angle_degs - car_angle_degs
            if heading_to_target < -180:
                heading_to_target = heading_to_target + 360
            
            # 3. calculate normalized efficiency of last move
            # vs. target acquisition
            dt = self.last_target_dist - target_dist
            
            if abs(dt) >= 12:
                dt = np.mean(self.target_deltas)
            
            # postive distance delta indicates "closing" on the target
            ndt = (dt- np.mean(self.target_deltas)) / np.std(self.target_deltas)
            
            # vs. obstacle avoidance
            do = min(sonar_dist_readings[:HUNT_NUM_SENSOR])
            
            # positive distance delta indicates "avoiding" an obstacle
            ndo = (do - np.mean(self.obstacle_dists)) / np.std(self.obstacle_dists)
            
            if cur_mode == ACQUIRE:
                move_efficiency = ndt / target_dist**0.333
                # cubed root of the target distance... lessens effect of distance
            else:
                move_efficiency = (ndt + (1.55 * ndo)) / target_dist**0.333
                # balancing avoidance with acquisition
                
            self.last_target_dist = target_dist
            self.target_deltas.append(dt)
            if len(self.target_deltas) > STATS_BUFFER:
                self.target_deltas.pop(0)
                                    
            self.last_obstacle_dist = min(sonar_dist_readings[:HUNT_NUM_SENSOR])
            self.obstacle_dists.append(do)
            if len(self.obstacle_dists) > STATS_BUFFER:
                self.obstacle_dists.pop(0)
            
            # 4. if w/in reasonable distance, declare victory
            if target_dist <= TARGET_RADIUS:
                print("************** target acquired ************")
                self.target_acquired = True
                #target_dist = 1
        
        if cur_mode == HUNT:
            hunt_readings = sonar_dist_readings[:HUNT_NUM_SENSOR]
            hunt_readings.append(target_dist)
            hunt_readings.append(heading_to_target)

        # build states
        turn_state = speed_state = acquire_state = hunt_state = 0
        turn_state = np.array([turn_readings])
        speed_state = np.array([speed_readings])

        if cur_mode in [ACQUIRE, HUNT]:
            acquire_state = np.array([[target_dist, heading_to_target]])
            if cur_mode == HUNT:
                hunt_state = np.array([hunt_readings])
    
        # calculate rewards based on training mode(s) in effect
        reward_turn = reward_speed = reward_acquire = reward_hunt = 0
        
        if cur_mode == SPEED:
            read = sonar_dist_readings[:SPEED_NUM_SENSOR]
        elif cur_mode == HUNT:
            read = sonar_dist_readings[:HUNT_NUM_SENSOR]
        else:
            read = sonar_dist_readings[:TURN_NUM_SENSOR]
        
        if self.car_is_crashed(read):
            # car crashed when any reading == 1. note: change (sensor) readings as needed
            self.crashed = True
            reward = -500
            if self.cur_x < 0 or self.cur_x > width or self.cur_y < 0 or self.cur_y > height:
                self.car_body.position = int(width/2), int(height/2)
                self.cur_x, self.cur_y = self.car_body.position
                self.num_off_scrn += 1
                print("off screen. total off screens", self.num_off_scrn)
                reward = -1000
            self.recover_from_crash(driving_direction)
            
        else:
            if cur_mode == TURN: # Rewards better spacing from objects
                reward = reward_turn = min(sonar_dist_readings)

            elif cur_mode == SPEED: # rewards distance from objects and speed
                reward = reward_speed = min(sonar_dist_readings)
                #sd_speeds = np.std(SPEEDS)
                #sd_dist = np.std(range(20))
            
                #std_speed = cur_speed / sd_speeds
                #std_dist = min(sonar_dist_readings[:TURN_NUM_SENSOR]) / sd_dist
            
                #std_max_speed = max(SPEEDS) / sd_speeds
                #std_max_dist = SONAR_ARM_LEN / sd_dist
            
                #reward_speed = ((std_speed * std_dist) +
                #               ((std_max_speed - std_speed) * (std_max_dist - std_dist)))

            elif cur_mode in [ACQUIRE, HUNT]: # rewards moving in the right direction and acquiring pixels
                if self.target_acquired == True:
                
                    reward = reward_acquire = reward_hunt = 1000
                
                    # remove acquired pixel
                    self.acquired_pixels.append(self.current_target)
                    self.target_pixels.remove(self.current_target)
                    print("pct complete:", (len(self.acquired_pixels) /
                                        (len(self.acquired_pixels) + len(self.target_pixels))))
                                        
                    if len(self.acquired_pixels) % 5 == 1:
                        take_screen_shot(screen)
                
                    self.assign_next_target((self.cur_x, self.cur_y), False)
                    self.target_acquired = False
            
                else:
                    if cur_mode == ACQUIRE:
                        reward = reward_acquire = 100 * move_efficiency
                    else:
                        if move_efficiency == 0:
                            reward_hunt = -2
                        else:
                            reward_hunt = 50 * move_efficiency
                        reward = reward_hunt

                if self.num_steps % 20000 == 0 or self.num_steps % 50000 == 1:
                    print("***** new rcd *****")
                    print("target dist:", target_dist)
                    print("dt:", dt)
                    print("mean dist deltas:", np.mean(self.target_deltas))
                    print("std dist deltas:", np.std(self.target_deltas))
                    print("ndt:", ndt)
                    print("min obs dist:", min(sonar_dist_readings[:HUNT_NUM_SENSOR]))
                    print("do:", do)
                    print("mean obs dists:", np.mean(self.obstacle_dists))
                    print("std obs dists:", np.std(self.obstacle_dists))
                    print("ndo:", ndo)
                    print("target dist ** 0.33:", target_dist**0.333)
                    print("move eff:", move_efficiency)
                    print("reward:", reward)
        
        self.num_steps += 1
        clock.tick()
        
        #if cur_speed != 70:
            #take_screen_shot(screen)
            #print(cur_speed)

        return turn_state, speed_state, acquire_state, hunt_state, reward, cur_speed, reward_turn, reward_acquire, reward_speed, reward_hunt
    
    # ***** turn and speed model functions *****
    
    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS[OBSTACLE_COLOR]
        self.space.add(c_body, c_shape)
        return c_body
    
    def create_cat(self,x,y):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        cat_body = pymunk.Body(1, inertia)
        cat_body.position = x, y
        cat_shape = pymunk.Circle(cat_body, 20)
        cat_shape.color = THECOLORS[CAT_COLOR]
        cat_shape.elasticity = 1.0
        cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(cat_body.angle)
        self.space.add(cat_body, cat_shape)
        return cat_body
    
    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, CAR_BODY_DIAM) # was 25
        self.car_shape.color = THECOLORS[CAR_COLOR]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)
    
    def move_obstacles(self):
        # randomly moves large, slow obstacles around
        for obstacle in self.obstacles:
            speed = random.randint(10, 15)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cats(self):
        # randomly moves small, fast obstacles
        for cat in self.cats:
            speed = random.randint(60, 80)
            direction = Vec2d(1, 0).rotated(random.randint(-3, 3)) # -2,2
            cat.velocity = speed * direction
            x, y = cat.position
            if x < 0 or x > width or y < 0 or y > height:
                cat.position = int(width/2), int(height/2)
    
    def car_is_crashed(self, sonar_dist_readings):
        return_val = False
        for reading in sonar_dist_readings:
            if reading <= 1:
                return_val = True
        return return_val

    def recover_from_crash(self, driving_direction):
        """
        we hit something, so recover
        """
        while self.crashed:
            crash_adjust = -100
            # Go backwards.
            self.car_body.velocity = crash_adjust * driving_direction # was -100
            self.crashed = False
            for i in range(10): # was 10
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def get_sonar_dist_color_readings(self, x, y, angle):
        sonar_dist_readings = []
        sonar_color_readings = []
        """
        sonar readings return N "distance" readings, one for each sonar. distance is
        a count of the first non-zero color detection reading starting at the object.
        """
        
        # make sonar "arms"
        arm_1 = self.make_sonar_arm(x, y)
        arm_2 = arm_1
        arm_3 = arm_1
        arm_4 = arm_1
        arm_5 = arm_1
        arm_6 = arm_1
        arm_7 = arm_1
        
        # rotate arms to get vector of readings
        d, c = self.get_arm_dist_color(arm_1, x, y, angle, 0)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_2, x, y, angle, 0.6)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_3, x, y, angle, -0.6)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_4, x, y, angle, 1.2)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_5, x, y, angle, -1.2)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_6, x, y, angle, 2.8)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_7, x, y, angle, -2.8)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        
        if show_sensors:
            pygame.display.update()

        return sonar_dist_readings, sonar_color_readings

    def get_arm_dist_color(self, arm, x, y, angle, offset):
        # count arm length to nearest obstruction
        i = 0

        # evaluate each arm point to see if we've hit something
        for point in arm:
            i += 1
            
            # move the point to the right spot
            rotated_p = self.get_rotated_point(x, y, point[0], point[1], angle + offset)
            
            # return i if rotated point is off screen
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 or rotated_p[0] >= width or rotated_p[1] >= height:
                return i, 1 # 1 is wall color
            else:
                obs = screen.get_at(rotated_p)
                # this gets the color of the pixel at the rotated point
                obs_color = self.get_track_or_not(obs)
                if obs_color != 0:
                    # if pixel not a safe color, return distance
                    return i, obs_color

            # plots the individual sonar point on the screen
            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        return i, 0 # 0 is safe color

    def make_sonar_arm(self, x, y):
        spread = 10  # gap between points on sonar arm
        distance = 10  # number of points on sonar arm
        arm_points = []
        # builds arm flat. it will be rotated about the center later
        for i in range(1, SONAR_ARM_LEN): # was 40
            arm_points.append((x + distance + (spread * i), y))

        return arm_points
    
    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        # check to see if color encountered is safe (i.e., should not be crash)
        if reading == pygame.color.THECOLORS[BACK_COLOR] or \
            reading == pygame.color.THECOLORS[CAR_COLOR] or \
            reading == pygame.color.THECOLORS[SMILE_COLOR] or \
            reading == pygame.color.THECOLORS[PATH_COLOR] or \
            reading == pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR] or \
            reading == pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR]:
            return 0
        else:
            if reading == pygame.color.THECOLORS[WALL_COLOR]:
                return 1
            elif reading == pygame.color.THECOLORS[CAT_COLOR]:
                return 2
            elif reading == pygame.color.THECOLORS[OBSTACLE_COLOR]:
                return 3

    # ***** target and acquire model functions *****

    def generate_targets(self, first_iter):
        
        # calculate number of targets that can fit space
        num_pxl_x_dir = int((width - 2 * ACQUIRE_MARGIN)/ACQUIRE_PIXEL_SEPARATION)
        num_pxl_y_dir = int((height- 2 * ACQUIRE_MARGIN)/ACQUIRE_PIXEL_SEPARATION)
        
        n = num_pxl_x_dir * num_pxl_y_dir
        
        ctr = 0
        for v in range(num_pxl_y_dir):
            for h in range(num_pxl_x_dir):
                
                # space targets across target grid
                x_pxl = (ACQUIRE_MARGIN + (h * ACQUIRE_PIXEL_SEPARATION))
                y_pxl = (ACQUIRE_MARGIN + (v * ACQUIRE_PIXEL_SEPARATION))
                
                if(first_iter == True):
                    self.target_pixels.append((x_pxl,y_pxl))
                
                ctr += 1

        return num_pxl_x_dir, num_pxl_y_dir

    def assign_next_target(self, last_xy, first_iter):
        
        # clear the path surface
        path_grid.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        # randomly select a new target
        if first_iter == False:
            
            pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR],
                             ((last_xy[0], height-last_xy[1]), (ACQUIRE_PIXEL_SIZE,
                                                         ACQUIRE_PIXEL_SIZE)), 0)
        
        self.current_target = random.choice(self.target_pixels)
        
        # closest logic
        #dx = np.array([int(i[0]) for i in self.target_pixels]) - last_xy[0]
        #dy = np.array([int(i[1]) for i in self.target_pixels]) - last_xy[1]
        #dists = np.sqrt(pow(dx,2) + pow(dy,2))
        #dists = dists.tolist()
        #index = dists.index(min(dists))
        #self.current_target = self.target_pixels[index]
        
        # draw the new target
        pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR],
                         ((self.current_target[0], height-self.current_target[1]),
                          (ACQUIRE_PIXEL_SIZE, ACQUIRE_PIXEL_SIZE)), 0)
                          
        print("new target:", self.current_target)


    # not currently used
    def get_touched_pixels(self,x,y):
        
        touched_pixels = []
        
        # use start and end points for last move to calculate line
        if self.last_xy[0] - x == 0:
            
            # have to take negative slope due to banged up way pygame reads screen
            m = -((y - self.last_xy[1]) / ((x+0.0001) - self.last_xy[0]))
            # use small x diff value to eliminate inf on vertical moves
            
        else:
            
            m = -((y - self.last_xy[1]) / (x - self.last_xy[0]))
    
        b = (y - (m * x))
            
        lo2hi_x = sorted([x,self.last_xy[0]])
        lo2hi_y = sorted([y,self.last_xy[1]])
            
        if len(range(lo2hi_x[0],lo2hi_x[1])) > 1:
            
            x_pixels = list(range(lo2hi_x[0]+1,lo2hi_x[1]+1))
        
        else:
            
            x_pixels = lo2hi_x
        
        if len(range(lo2hi_y[0],lo2hi_y[1])) > 1:
            
            y_pixels = list(range(lo2hi_y[0]+1,lo2hi_y[1]+1))
                
        else:
            
            y_pixels = lo2hi_y

        # save any points that overlap target pixels
        for i in range(len(x_pixels)):
            for j in range(len(y_pixels)):
                if y_pixels[j] == m * x_pixels[i] + b:
                    touched_pixels.append((x_pixels[i],y_pixels[j]))

        return touched_pixels

# ***** global functions *****

def take_screen_shot(screen):
    time_taken = time.asctime(time.localtime(time.time()))
    time_taken = time_taken.replace(" ", "_")
    time_taken = time_taken.replace(":",".")
    save_file = "screenshots/" + time_taken + ".jpeg"
    pygame.image.save(screen,save_file)
    print("screen shot taken")

# HOLD FOR CONVNET: down sample the image
# pathSurf_scaled = pygame.transform.smoothscale(pathSurf,
#                                               (int(width/DOWN_SAMPLE),
#                                                int(height/DOWN_SAMPLE)))
# convert the down-sampled screen surface into a 2D array of floating point values, normalized to have approximately zero mean and standard deviation ~0.5.
#path_string = pygame.image.tostring(pathSurf_scaled, "RGB")
#pixel_depth = 255.0
#tmp_file = 'tmp.jpg'
#pygame.image.save(pathSurf_scaled,tmp_file)
#im = ndimage.imread(tmp_file).astype(float)
#im = (im - (pixel_depth/2)) / pixel_depth
# no real value in de-meaning as they're all black or grey
#im = np.mean(im, axis=2)
#cur_path = im.flatten("F")

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
