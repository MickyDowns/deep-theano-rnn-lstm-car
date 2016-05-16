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
TURN = 1
SPEED = 2
ACQUIRE = 3
HUNT = 4
PACK = 5
cur_mode = SPEED

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
DRONE_COLOR = "green"
SMILE_COLOR = "blue"
OBSTACLE_COLOR = "purple"
CAT_COLOR = "orange"
DRONE_BODY_DIAM = 12
SONAR_ARM_LEN = 20
show_sensors = True # Showing sensors and redrawing slows things down.
draw_screen = True

# turn model settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings

# speed model settings
SPEED_NUM_SENSOR = 7 # front five, rear two sonar
SPEEDS = [30,50,70]

# acquire model settngs
target_grid = pygame.Surface((width, height), pygame.SRCALPHA, 32)
target_grid.convert_alpha()
ACQUIRE_PIXEL_COLOR = "green"
ACQUIRED_PIXEL_COLOR = "yellow"
ACQUIRE_PIXEL_SIZE = 2

if cur_mode == ACQUIRE:
    ACQUIRE_PIXEL_SEPARATION = 25
    ACQUIRE_MARGIN = 50
else:
    ACQUIRE_PIXEL_SEPARATION = 5
    ACQUIRE_MARGIN = 75

path_grid = pygame.Surface((width, height))
PATH_COLORS = ["light grey", "grey", "dark grey"]

# hunt model settings
HUNT_NUM_SENSOR = 7
STATS_BUFFER = 2000

# pack model settings
NUM_DRONES = 2
PACK_HEADING_ADJUST = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
NUM_TARGETS = 1

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# ***** instantiate the game *****

class GameState:
    def __init__(self):
        # initialize space
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        
        # initialize counters
        self.num_steps = 0
        self.num_off_scrn = 0
        
        # crash state
        self.crashed = np.empty([NUM_DRONES, 1])
        self.crashed.fill(False)

        # create drones
        self.drones = []
        self.cur_x = self.cur_y = np.zeros([NUM_DRONES, 1])
        for drone_id in NUM_DRONES:
            self.drones.append(self.create_drone(width/2, height/2, 0.5))
            self.cur_x[drone_id], self.cur_y[drone_id] = self.drones[drone_id].body.position

        # create walls
        static = [pymunk.Segment(self.space.static_body,(0, 1), (0, height), 1),
                  pymunk.Segment(self.space.static_body,(1, height), (width, height), 1),
                  pymunk.Segment(self.space.static_body,(width-1, height), (width-1, 1), 1),
                  pymunk.Segment(self.space.static_body,(1, 1), (width, 1), 1)]
        
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS[WALL_COLOR]
        self.space.add(static)
        
        if cur_mode in [TURN, SPEED, HUNT]:
            
            self.obstacles = self.cats = []
            
            #if cur_mode in [TURN, AVOID]: # used to gradually introduce obstacles
            # create slow, randomly moving, larger obstacles
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-100),50))
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-70),50))
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-70),63))
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-70),63))
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-70),30))
            self.obstacles.append(self.create_obstacle(random.randint(100, width-100),
                                                       random.randint(70, height-70),30))
        
            # create faster, randomly moving, smaller obstacles a.k.a. "cats"
            self.cats.append(self.create_cat(width-950,height-100))
            self.cats.append(self.create_cat(width-50,height-600))
            self.cats.append(self.create_cat(width-50,height-100))
            self.cats.append(self.create_cat(width-50,height-600))

        if cur_mode in [ACQUIRE, HUNT, PACK]:
            # initialize last position values
            self.last_x = np.empty([NUM_DRONES,1]); self.last_x.fill(width/2 + 1)
            self.last_y = np.empty(NUM_DRONES,1); self.last_y.fill(height/2 +1)
            
            # set up seach grid and feed first target
            self.target_inventory = self.acquired_targets = []
            self.target_radius = 5
            self.generate_targets(True)
            self.current_targets = []
            for targets in NUM_TARGETS:
                self.assign_target((self.last_x[0], self.last_y[0]), True)
            self.target_acquired = False
            
            # initialize target distances
            self.last_tgt_dist = np.empty([NUM_DRONES, NUM_TARGETS])
            self.last_tgt_dist.fill(350)
            self.tgt_deltas = np.empty([NUM_DRONES, NUM_TARGETS])
            self.tgt_deltas.fill(0.5)
            
            # initialize obstacle distances
            self.last_obs_dist = np.empty([NUM_DRONES, NUM_TARGETS])
            self.last_obs_dist.fill(10)
            self.obs_dists = np.empty([NUM_DRONES, NUM_TARGETS])
            self.tgt_dists.fill(10)
        
        """ note: while python assumes (0,0) is in the lower left of the screen, pygame assumes (0,0) in the upper left. therefore, y+ moves DOWN the y axis. here is example code that illustrates how to handle angles in that environment: https://github.com/mgold/Python-snippets/blob/master/pygame_angles.py. in this implementation, I have flipped the screen so that Y+ moves UP the screen."""

    # ***** primary logic controller for game play *****

    def frame_step(self, cur_mode, drone_id, turn_action, speed_action, pack_action, cur_speed, cum_distance):
        
        # turn drone based on current (active) model prediction
        if cur_mode in [TURN, AVOID, ACQUIRE, HUNT, PACK]:
            self.set_turn(turn_action, self.drones[drone_id].body)
        
        # set speed based on active model prediction
        if cur_mode in [AVOID, HUNT, PACK]: # setting speed values directly see SPEEDS
            self.set_speed(cur_speed, speed_action, self.drones[drone_id].body)
    
        # move obstacles
        if cur_mode in [TURN, SPEED, HUNT, PACK]:
            # slow obstacles
            if self.num_steps % 20 == 0: # 20x slower than self
                self.move_obstacles()

            # fast obstacles
            if self.num_steps % 40 == 0: # 40 x more stable than self
                self.move_cats
        
        # adjust heading based on pack model output
        if cur_mode == PACK:
            pack_heading_adjust = self.set_pack_adjust(pack_action)

        # update the screen and surfaces
        if drone_id == 0:
            screen.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        if cur_mode in [ACQUIRE, HUNT, PACK]:
            # draw the path drone has taken on the path grid
            pygame.draw.lines(path_grid, pygame.color.THECOLORS[PATH_COLORS[drone_id]], True,
                              ((self.last_x[drone_id], height - self.last_y[drone_id]),
                               (self.cur_x[drone_id], height - self.cur_y[drone_id])), 1)
            
            # if last drone, bind paths, targets to the screen
            if drone_id == (NUM_DRONES - 1):
                screen.blit(path_grid, (0,0))
                screen.blit(target_grid, (0,0))

        # if last drone, display screen
        if(drone_id == (NUM_DRONES - 1)):
            draw(screen, self.space)
            self.space.step(1./10) # one pixel for every 10 SPEED
            if draw_screen:
                pygame.display.flip()

        # get readings, build states
        self.last_x[drone_id] = self.cur_x[drone_id]
        self.last_y[drone_id] = self.cur_y[drone_id]
        self.cur_x[drone_id], self.cur_y[drone_id] = self.drones[drone_id].body.position

        turn_state, avoid_state, acquire_state, hunt_state, min_sonar_dist, move_efficiency = \
            self.build_states(cur_mode, drone_id, turn_action, cur_speed)

        # calculate rewards based on training mode(s) in effect
        turn_reward, speed_reward, acquire_reward, hunt_reward, drone_reward = \
            self.calculate_reward(cur_mode, cur_x, cur_y, self.c1_body,
                                  min_sonar_dist, move_efficiency)

        reward = max([turn_reward, speed_reward, acquire_reward, hunt_reward])

        if cur_mode == PACK:
            c2_turn_reward, c2_speed_reward, c2_acquire_reward, c2_hunt_reward, c2_reward_pack = \
                self.calculate_reward(cur_mode, c2_cur_x, c2_cur_y, self.c2_body,
                                      c2_min_sonar_dist, c2_move_efficiency)
            
            reward = c1_hunt_reward + c2_hunt_reward
        
        self.num_steps += 1
        clock.tick()
        
        #if c1_cur_speed != 70:
            #take_screen_shot(screen)
            #print(c1_cur_speed)

        return turn_state, avoid_state, acquire_state, hunt_state, drone_state, reward, cur_speed
                
    
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
    
    
    def create_drone(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        drone_body = pymunk.Body(1, inertia)
        drone_body.position = x, y
        drone_shape = pymunk.Circle(drone_body, DRONE_BODY_DIAM) # was 25
        drone_shape.color = THECOLORS[DRONE_COLOR]
        drone_shape.elasticity = 1.0
        drone_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(drone_body.angle)
        drone_body.apply_impulse(driving_direction)
        self.space.add(drone_body, drone_shape)
        return drone_body
    
    
    def move_obstacles(self):
        # randomly moves large, slow obstacles around
        for obstacle in self.obstacles:
            speed = random.randint(10, 15)
            direction = Vec2d(1, 0).rotated(self.c1_body.angle + random.randint(-2, 2))
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


    def set_turn(self, turn_action, body):
        # action == 0 is continue current trajectory
        if turn_action == 1:  # slight right adjust to current trajectory
            body.angle -= .2
        elif turn_action == 2:  # hard right
            body.angle -= .4
        elif turn_action == 3:  # slight left
            body.angle += .2
        elif turn_action == 4:  # hard left
            body.angle += .4


    def set_speed(self, cur_speed, speed_action, body):
        # choose appropriate speed action, including 0 speed
        if speed_action == 0:
            set_speed = SPEEDS[0]
        elif speed_action == 1:
            set_speed = SPEEDS[1]
        elif speed_action == 2:
            set_speed = SPEEDS[2]
        
        # effect move by applying speed and direction as vector on self
        driving_direction = Vec2d(1, 0).rotated(body.angle)
        body.velocity = cur_speed * driving_direction


    def set_pack_adjust(self, pack_action):
        # action == 0 is continue current trajectory
        if turn_action == 1:  # +45 degree heading adjustment
            pack_heading_adjust = (3.14 / 4) * 1
        elif turn_action == 2:  # +90 degree heading adjustment
            pack_heading_adjust = (3.14 / 4) * 2
        elif turn_action == 3:  # -45 degree
            pack_heading_adjust = (3.14 / 4) * -1
        elif turn_action == 4:  # -90 degree
            pack_heading_adjust = (3.14 / 4) * -2
                
        return pack_heading_adjust


    def evaluate_move(self, cur_mode, drone_id, pack_htt_adjust):
        # 1. calculate distance and angle to active target(s)
        # a. euclidean distance traveled
        cur_x, cur_y = self.drones[drone_id].body.position
        dx = self.current_targets[0] - cur_x
        dy = self.current_targets[1] - cur_y
            
        target_dist = ((dx**2 + dy**2)**0.5)
            
        # b. calculate target angle
        # i. relative to drone
        rads = atan2(dy,dx)
        rads %= 2*pi
        target_angle_degs = degrees(rads)
            
        if target_angle_degs > 180:
            target_angle_degs = target_angle_degs - 360
            
        # ii. relative to drone's current direction
        rads = body.angle
        rads %= 2*pi
        drone_angle_degs = degrees(rads)
            
        if drone_angle_degs > 360:
            drone_angle_degs = drone_angle_degs - 360
            
        # "heading" accounts for angle from drone and of drone netting degrees drone must turn
        heading_to_target = target_angle_degs - drone_angle_degs
        if heading_to_target < -180:
            heading_to_target = heading_to_target + 360

        # 3. calculate normalized efficiency of last move
        # vs. target acquisition
        dt = self.last_tgt_dist[drone_id] - target_dist
            
        if abs(dt) >= 12:
            dt = np.mean(self.tgt_deltas[drone_id,:])

        # postive distance delta indicates "closing" on the target
        ndt = (dt- np.mean(self.tgt_deltas[drone_id,:])) / np.std(tgt_deltas[drone_id,:])
    
        # vs. obstacle avoidance
        do = min(sonar_dist_readings[:HUNT_NUM_SENSOR])
        
        # positive distance delta indicates "avoiding" an obstacle
        ndo = (do - np.mean(self.obs_dists[drone_id,:])) / np.std(obs_dists[drone_id,:])
            
        if cur_mode == ACQUIRE:
            acquire_move_efficiency = ndt / target_dist**0.333
            # cubed root of the target distance... lessens effect of distance
        else:
            avoid_move_efficiency = ndo / target_dist**0.333
            acquire_move_efficiency = ndt / target_dist**0.333
            # for balancing avoidance with acquisition
            
        self.last_tgt_dist[drone_id] = target_dist
        if drone_id == 0:
            self.tgt_deltas[drone_id].append(dt)
        else:
            self.tgt_deltas[drone_id,self.tgt_deltas.shape[1]] = dt

        if len(tgt_deltas) > STATS_BUFFER:
            tgt_deltas.pop(0)
            
        last_obs_dist = min(sonar_dist_readings[:HUNT_NUM_SENSOR])
        obs_dists.append(do)
        if len(obs_dists) > STATS_BUFFER:
            obs_dists.pop(0)
            
        # 4. if w/in reasonable distance, declare victory
        if target_dist <= TARGET_RADIUS:
            print("************** target acquired ************")
            self.target_acquired = True
        
            # move acquired target to acquired targets
            self.acquired_targets.append(self.current_targets[target_id])
            self.target_inventory.remove(self.current_targets[target_id])
            
            print("pct complete:", (len(self.acquired_targets) /
                                    (len(self.acquired_targets) + len(self.target_inventory))))
                    
            if len(self.acquired_targets) % 5 == 1:
                take_screen_shot(screen)
            
            # get a new target
            self.assign_target((cur_x, cur_y), False)

        return target_dist, heading_to_target, move_efficiency


    def build_states(self, cur_mode, drone_id, turn_action, cur_speed):
        
        turn_state = avoid_state = acquire_state = hunt_state = move_efficiency = 0
    
        # get readings from the various sensors
        sonar_dist_readings, sonar_color_readings = \
            self.get_sonar_dist_color_readings(self.cur_x[drone_id],
                                               self.cur_y[drone_id],
                                               self.drones[drone_id].body.angle)
        
        turn_readings = sonar_dist_readings[:TURN_NUM_SENSOR]
        turn_readings = turn_readings + sonar_color_readings[:TURN_NUM_SENSOR]
        
        avoid_readings = sonar_dist_readings[:AVOID_NUM_SENSOR]
        avoid_readings = avoid_readings + sonar_color_readings[:AVOID_NUM_SENSOR]
        avoid_readings.append(turn_action)
        avoid_readings.append(cur_speed)
        
        if cur_mode in [ACQUIRE, HUNT, PACK]:
            # calculate distances, headings and efficiency
            target_dist, heading_to_target, move_efficiency = \
                self.evaluate_move(cur_mode, dron_id)
            
            acquire_state = np.array([[target_dist, heading_to_target]])
            # IS SONAR DISTANCE READING USED HERE? HASN'T IT BEEN CRASHING. HOW HAS IT BEEN CRASHING?

            if cur_mode in [HUNT, PACK]:
                hunt_readings = sonar_dist_readings[:HUNT_NUM_SENSOR]
                min_sonar_dist = min(hunt_readings)
                hunt_readings.append(target_dist)
                hunt_readings.append(heading_to_target)
                hunt_state = np.array([hunt_readings])

        return turn_state, avoid_state, acquire_state, hunt_state, min_sonar_dist, move_efficiency


    def calculate_reward(self, mode, cur_x, cur_y, body, min_sonar_dist, move_efficiency)

        turn_reward = speed_reward = acquire_reward = hunt_reward = 0
    
        # check for crash
        if min_sonar_dist <= 1:
            reward = -500
            if cur_x < 0 or cur_x > width or cur_y < 0 or cur_y > height:
                body.position = int(width/2), int(height/2)
                cur_x, cur_y = body.position
                self.num_off_scrn += 1
                print("off screen. total off screens", self.num_off_scrn)
                reward = -1000
            self.recover_from_crash(driving_direction, body)
        
        else:
            if cur_mode == TURN: # Rewards better spacing from objects
                turn_reward = min_sonar_dist
            
            elif cur_mode == SPEED: # rewards distance from objects and speed
                speed_reward = min_sonar_dist
            #sd_speeds = np.std(SPEEDS)
            #sd_dist = np.std(range(20))
            
            #std_speed = c1_cur_speed / sd_speeds
            #std_dist = min(c1_sonar_dist_readings[:TURN_NUM_SENSOR]) / sd_dist
            
            #std_max_speed = max(SPEEDS) / sd_speeds
            #std_max_dist = SONAR_ARM_LEN / sd_dist
            
            #speed_reward = ((std_speed * std_dist) +
            #               ((std_max_speed - std_speed) * (std_max_dist - std_dist)))
            
            elif cur_mode in [ACQUIRE, HUNT, PACK]: # rewards moving in the right direction and acquiring pixels
                if self.target_acquired == True:
                    acquire_reward = hunt_reward = 1000
                    self.target_acquired = False

                else:
                    if cur_mode == ACQUIRE:
                        acquire_reward = 100 * move_efficiency
                    else:
                        if move_efficiency == 0:
                            hunt_reward = -2
                        else:
                            hunt_reward = 50 * move_efficiency
            
            #if self.num_steps % 50000 == 0 or self.num_steps % 50000 == 1:
            #    print("***** new rcd *****")
            #    print("target dist:", target_dist)
            #    print("dt:", dt)
            #    print("mean dist deltas:", np.mean(self.c1_tgt_deltas))
            #    print("std dist deltas:", np.std(self.c1_tgt_deltas))
            #    print("ndt:", ndt)
            #    print("min obs dist:", min(c1_sonar_dist_readings[:HUNT_NUM_SENSOR]))
            #    print("do:", do)
            #    print("mean obs dists:", np.mean(self.c1_obs_dists))
            #    print("std obs dists:", np.std(self.c1_obs_dists))
            #    print("ndo:", ndo)
            #    print("target dist ** 0.33:", target_dist**0.333)
            #    print("move eff:", move_efficiency)
            #    print("reward:", reward)
            
            return turn_reward, speed_reward, acquire_reward, hunt_reward


    def recover_from_crash(self, driving_direction, body):
        """
        we hit something, so recover
        """
            #while self.crashed:
            crash_adjust = -100
            # back up
            body.velocity = crash_adjust * driving_direction
            #self.crashed = False
            for i in range(10): # was 10
                body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()


    def get_sonar_dist_color_readings(self, x, y, angle):
        c1_sonar_dist_readings = []
        c1_sonar_color_readings = []
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
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_2, x, y, angle, 0.6)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_3, x, y, angle, -0.6)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_4, x, y, angle, 1.2)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_5, x, y, angle, -1.2)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_6, x, y, angle, 2.8)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_7, x, y, angle, -2.8)
        c1_sonar_dist_readings.append(d); c1_sonar_color_readings.append(c)
        
        if show_sensors:
            pygame.display.update()

        return c1_sonar_dist_readings, c1_sonar_color_readings


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
            reading == pygame.color.THECOLORS[DRONE_COLOR] or \
            reading == pygame.color.THECOLORS[SMILE_COLOR] or \
            reading == pygame.color.THECOLORS[PATH_COLOR_1] or \
            reading == pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR] or \
            reading == pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR] or \
            reading == pygame.color.THECOLORS[PATH_COLOR_2]:
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
                    self.target_inventory.append((x_pxl,y_pxl))
                
                ctr += 1

        return num_pxl_x_dir, num_pxl_y_dir

    def assign_target(self, last_xy, first_iter, old_target_xy):
        
        # clear the path surface
        path_grid.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        # randomly select a new target
        if !first_iter:
            
            pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR],
                             ((old_target_xy[0], height-old_target_xy[1]),
                              (ACQUIRE_PIXEL_SIZE, ACQUIRE_PIXEL_SIZE)), 0)
        
        new_target = random.choice(self.target_inventory)
        # closest logic
        #dx = np.array([int(i[0]) for i in self.target_inventory]) - last_xy[0]
        #dy = np.array([int(i[1]) for i in self.target_inventory]) - last_xy[1]
        #dists = np.sqrt(pow(dx,2) + pow(dy,2))
        #dists = dists.tolist()
        #index = dists.index(min(dists))
        #self.current_targets = self.target_inventory[index]
        
        self.current_targets.append(new_target)
        print("new target:", new_target)
        
        # draw the new target
        pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR],
                         ((self.current_targets[0], height-self.current_targets[1]),
                          (ACQUIRE_PIXEL_SIZE, ACQUIRE_PIXEL_SIZE)), 0)


    # not currently used
    def get_touched_pixels(self,x,y):
        
        touched_pixels = []
        
        # use start and end points for last move to calculate line
        if self.c1_last_xy[0] - x == 0:
            
            # have to take negative slope due to banged up way pygame reads screen
            m = -((y - self.c1_last_xy[1]) / ((x+0.0001) - self.c1_last_xy[0]))
            # use small x diff value to eliminate inf on vertical moves
            
        else:
            
            m = -((y - self.c1_last_xy[1]) / (x - self.c1_last_xy[0]))
    
        b = (y - (m * x))
            
        lo2hi_x = sorted([x,self.c1_last_xy[0]])
        lo2hi_y = sorted([y,self.c1_last_xy[1]])
            
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
