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
AVOID = 2
ACQUIRE = 3
HUNT = 4
PACK = 5
cur_mode = PACK

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
OBSTACLE_SIZES = [30, 30, 50, 50, 63, 63]
show_sensors = True # Showing sensors and redrawing slows things down.
draw_screen = True

# turn model settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings

# avoid model settings
AVOID_NUM_SENSOR = 7 # front five, rear two sonar
SPEEDS = [30,50,70]

# acquire model settngs
target_grid = pygame.Surface((width, height), pygame.SRCALPHA, 32)
target_grid.convert_alpha()
ACQUIRE_PIXEL_COLOR = "green"
ACQUIRED_PIXEL_COLOR = "yellow"
ACQUIRE_PIXEL_SIZE = 2
TARGET_RADIUS = 10

if cur_mode == ACQUIRE:
    ACQUIRE_PIXEL_SEPARATION = 25
    ACQUIRE_MARGIN = 50
else:
    ACQUIRE_PIXEL_SEPARATION = 5
    ACQUIRE_MARGIN = 75

path_grid = pygame.Surface((width, height))
PATH_COLOR = "grey"

# hunt model settings
HUNT_NUM_SENSOR = 7 # all sonar distance / color readings
DRONE_NUM_SENSOR = 7 # all sonar distance readings
NUM_DRONES = 2
PACK_HEADING_ADJUST = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]
NUM_TARGETS = 1
PACK_EVAL_FRAMES = 5
STATS_BUFFER = 2000

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# ***** instantiate the game *****

class GameState:
    def __init__(self):
        # initialize space
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        
        # initialize counters
        self.total_frame_ctr = 0
        self.replay_frame_ctr = 0
        self.acquire_frame_ctr = 0
        self.num_off_scrn = 0
        
        # create drones
        self.drones = []
        
        for drone_id in range(NUM_DRONES):
            self.drones.append(self.create_drone(random.randint(400,600),
                                                 random.randint(300,400), 0.5))
        
        self.last_x = np.empty([NUM_DRONES,1])
        self.last_y = np.empty([NUM_DRONES,1])
        for drone_id in range(len(self.drones)):
            x, y = self.drones[drone_id].position
            self.last_x[drone_id] = x + 2; self.last_y[drone_id] = y + 2
        
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
        
        if cur_mode in [TURN, AVOID, HUNT, PACK]:
            
            self.obstacles = []
            self.cats = []
            
            if cur_mode in [TURN, AVOID]: # used to gradually introduce obstacles
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
            
            # set up seach grid and feed first target
            self.target_inventory = []
            self.acquired_targets = []
            self.current_targets = []
            self.target_radius = TARGET_RADIUS
            self.generate_targets(True)
            for targets in range(NUM_TARGETS):
                self.assign_target(True, drone_id)
            self.target_acquired = False
            
            # initialize structures to track efficience of EACH move
            # distance to target
            self.last_tgt_dist = np.empty([NUM_DRONES, NUM_TARGETS])
            self.last_tgt_dist.fill(350) # last target dist held in array for ea drone
            tmp = [0.5, 1]; self.tgt_deltas = [] # deltas held in list for ea drone
            for drone_id in range(NUM_DRONES): self.tgt_deltas.append(tmp)
            
            # distance to obstacles
            self.last_obs_dist = np.empty([NUM_DRONES, NUM_TARGETS])
            self.last_obs_dist.fill(10) # last obstacle dist held in array for ea drone
            tmp = [10, 12]; self.obs_dists = [] # distances held in list for ea drone
            for drone_id in range(NUM_DRONES): self.obs_dists.append(tmp)

            # initialize structures to track efficience of PACK_EVAL_FRAMES moves
            if cur_mode == PACK:
                # distance to target
                self.last_pack_tgt_dist = np.empty([NUM_DRONES, NUM_TARGETS])
                self.last_pack_tgt_dist.fill(350) # last target dist held in array for ea drone
                tmp = [0.4, 0.7]; self.pack_tgt_deltas = [] # deltas held in list for ea drone
                for drone_id in range(NUM_DRONES): self.pack_tgt_deltas.append(tmp)
        
                # distance to obstacles
                self.last_pack_obs_dist = np.empty([NUM_DRONES, NUM_TARGETS])
                self.last_pack_obs_dist.fill(10) # last obstacle dist held in array for ea drone
                tmp = [10, 12]; self.pack_obs_dists = [] # distances held in list for ea drone
                for drone_id in range(NUM_DRONES): self.pack_obs_dists.append(tmp)

                # starting positions
                self.pack_cum_rwds = np.zeros([NUM_DRONES, 1])
                self.start_positions = [(25,25), (25,675), (975,25), (975,650)]
                self.start_angles = [0.8, -0.8, 2.5, 3.9]

        """ note: while python assumes (0,0) is in the lower left of the screen, pygame assumes (0,0) in the upper left. therefore, y+ moves DOWN the y axis. here is example code that illustrates how to handle angles in that environment: https://github.com/mgold/Python-snippets/blob/master/pygame_angles.py. in this implementation, I have flipped the screen so that Y+ moves UP the screen."""

    # ***** primary logic controller for game play *****

    def frame_step(self, drone_id, turn_action, speed_action, pack_action, cur_speed, total_ctr, replay_ctr):

        self.total_frame_ctr = total_ctr
        self.replay_frame_ctr = replay_ctr
        self.acquire_frame_ctr += 1
        
        # turn drone based on current (active) model prediction
        if cur_mode in [TURN, AVOID, ACQUIRE, HUNT, PACK]:
            self.set_turn(turn_action, drone_id)
    
        # set speed based on active model prediction
        if cur_mode in [AVOID, HUNT, PACK]: # setting speed values directly see SPEEDS
            cur_speed = self.set_speed(speed_action, drone_id)
        
        # effect move by applying speed and direction as vector on self
        driving_direction = Vec2d(1, 0).rotated(self.drones[drone_id].angle)
        self.drones[drone_id].velocity = cur_speed * driving_direction
        x, y = self.drones[drone_id].position
        
        # set heading adjustment based on pack model output
        if cur_mode == PACK:
            heading_adjust = self.set_pack_adjust(pack_action)[drone_id]
        else:
            heading_adjust = 0
        
        # move obstacles
        if cur_mode in [TURN, AVOID, HUNT, PACK]:
            # slow obstacles
            if self.total_frame_ctr % 20 == 0: # 20x slower than self
                self.move_obstacles()

            # fast obstacles
            if self.total_frame_ctr % 40 == 0: # 40 x more stable than self
                self.move_cats()
        
        # update the screen and surfaces
        if drone_id == 0:
            screen.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        if cur_mode in [ACQUIRE, HUNT, PACK]:
            # draw the path drone has taken on the path grid
            if self.acquire_frame_ctr / NUM_DRONES > 1.5:
                pygame.draw.lines(path_grid, pygame.color.THECOLORS[PATH_COLOR], True,
                                  ((self.last_x[drone_id], height - self.last_y[drone_id]),
                                   (x, height - y)), 1)
            
            # if last drone, bind paths, targets to the screen
            if drone_id == (NUM_DRONES - 1):
                screen.blit(path_grid, (0,0))
                screen.blit(target_grid, (0,0))

        # if last drone, display screen
        #if(drone_id == (NUM_DRONES - 1)):
        draw(screen, self.space)
        self.space.step(1./10) # one pixel for every 10 SPEED
        if draw_screen:
            pygame.display.flip()

        # get readings, build states
        self.last_x[drone_id] = x; self.last_y[drone_id] = y
        x, y = self.drones[drone_id].position

        turn_state, avoid_state, acquire_state, hunt_state, drone_state, min_sonar_dist, avoid_move_efficiency, acquire_move_efficiency = \
            self.build_states(drone_id, turn_action, heading_adjust, cur_speed)
        
        # calc rewards based on training mode(s) in effect
        reward = self.calc_rwd(drone_id, min_sonar_dist, driving_direction, cur_speed, avoid_move_efficiency, acquire_move_efficiency)
        
        # introduce obstacles gradually for HUNT/PACK learning
        if cur_mode in [HUNT, PACK] and drone_id == (NUM_DRONES - 1):
            if self.total_frame_ctr > 1 and \
                self.total_frame_ctr < 601 and \
                self.total_frame_ctr % 100 == 0:
                self.obstacles.append(self.create_obstacle(random.randint(200, width-200),
                                                           random.randint(140, height-140),
                                                           OBSTACLE_SIZES[int(self.total_frame_ctr / 100)-1]))
                self.target_radius -= 1
        
            if self.total_frame_ctr > 601 and \
                self.total_frame_ctr < 1001 and \
                self.total_frame_ctr % 100 == 0:

                self.cats.append(self.create_cat(width-500,height-350))

        #self.total_frame_ctr += 1
        clock.tick()

        return turn_state, avoid_state, acquire_state, hunt_state, drone_state, reward, cur_speed
                
    
    # ***** turn and speed model functions *****
    
    def create_obstacle(self, x, y, r):
        obs_body = pymunk.Body(pymunk.inf, pymunk.inf)
        obs_shape = pymunk.Circle(obs_body, r)
        obs_shape.elasticity = 1.0
        obs_body.position = x, y
        obs_shape.color = THECOLORS[OBSTACLE_COLOR]
        self.space.add(obs_body, obs_shape)
        return obs_body
    
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
        if len(self.obstacles) > 0:
            for obstacle in self.obstacles:
                speed = random.randint(10, 15)
                direction = Vec2d(1, 0).rotated(self.drones[0].angle + random.randint(-2, 2))
                obstacle.velocity = speed * direction
    
    def move_cats(self):
        # randomly moves small, fast obstacles
        if len(self.cats) > 0:
            for cat in self.cats:
                speed = random.randint(60, 80)
                direction = Vec2d(1, 0).rotated(random.randint(-3, 3)) # -2,2
                
                x, y = cat.position
                if x < 0 or x > width or y < 0 or y > height:
                    cat.position = int(width/2), int(height/2)
                cat.velocity = speed * direction


    def set_turn(self, turn_action, drone_id):
        # action == 0 is continue current trajectory
        if turn_action == 1:  # slight right adjust to current trajectory
            self.drones[drone_id].angle -= .2
        elif turn_action == 2:  # hard right
            self.drones[drone_id].angle -= .4
        elif turn_action == 3:  # slight left
            self.drones[drone_id].angle += .2
        elif turn_action == 4:  # hard left
            self.drones[drone_id].angle += .4


    def set_speed(self, speed_action, drone_id):
        # choose appropriate speed action, including 0 speed
        if speed_action == 0:
            cur_speed = SPEEDS[0]
        elif speed_action == 1:
            cur_speed = SPEEDS[1]
        elif speed_action == 2:
            cur_speed = SPEEDS[2]

        return cur_speed


    def set_pack_adjust(self, pack_action):
        
        heading_adjust = []
        # pack actions effect +/- 0.8 radian (45 deg) drone heading adjustments (2)
        for i in range(NUM_DRONES):
            heading = PACK_HEADING_ADJUST[pack_action][i]
            heading_adjust.append(heading * (3.14 / 4))
                
        return heading_adjust


    def evaluate_move(self, drone_id, heading_adjust, min_sonar_dist):
        
        #print("*** in eval ***")
        # eventually, want to introduce multiple targets. each new target doubles
        # state variables. so, for now, assuming single target:
        target_id = 0
        avoid_move_efficiency = 0
        acquire_move_efficiency = 0
        
        # 1. calc distance and angle to active target(s)
        # a. euclidean distance traveled
        x, y = self.drones[drone_id].position
        dx = self.current_targets[target_id][0] - x
        dy = self.current_targets[target_id][1] - y
        
        target_dist = int(((dx**2 + dy**2)**0.5))
        #print("target dist:", target_dist)
        # b. calc target angle
        # i. relative to drone
        rads = atan2(dy,dx)
        rads %= 2*pi
        
        true_target_angle_rads = np.round(rads,1)
        if true_target_angle_rads > 3.14:
            true_target_angle_rads = true_target_angle_rads - 6.28
        
        true_target_angle_degs = degrees(rads)
        if true_target_angle_degs > 180:
            true_target_angle_degs = true_target_angle_degs - 360
        #print("true angle degrees:", true_target_angle_degs)
        
        rads = atan2(dy,dx)
        rads = rads + heading_adjust
        rads %= 2*pi
        adj_target_angle_degs = degrees(rads)
        
        if adj_target_angle_degs > 180:
            adj_target_angle_degs = adj_target_angle_degs - 360
        #print("adj angle degrees:", adj_target_angle_degs)

        # ii. relative to drone's current direction
        rads = self.drones[drone_id].angle
        rads %= 2*pi
        drone_angle_rads = rads
        drone_angle_degs = degrees(rads)
            
        if drone_angle_degs > 360:
            drone_angle_degs = drone_angle_degs - 360
        #print("heading degrees:", drone_angle_degs)
            
        # "heading" accounts for angle FROM drone and OF drone netting degrees drone must turn
        adj_heading_to_target = adj_target_angle_degs - drone_angle_degs
        if adj_heading_to_target < -180:
            adj_heading_to_target = adj_heading_to_target + 360
        
        if cur_mode != PACK:
            # 3. calc normalized efficiency of last move
            # vs. target acquisition
            dt = int(self.last_tgt_dist[drone_id, target_id] - target_dist)
            
            if abs(dt) >= 12: # mistakenly thinking crashes are moves. so, ignore moves > 12
                dt = np.mean(self.tgt_deltas[drone_id])

            # postive distance delta indicates "closing" on the target
            ndt = np.round((dt - np.mean(self.tgt_deltas[drone_id])) / np.std(self.tgt_deltas[drone_id]),2)
            
            # save current values
            self.last_tgt_dist[drone_id, target_id] = target_dist
            self.tgt_deltas[drone_id].append(dt)

            if len(self.tgt_deltas[drone_id]) > STATS_BUFFER:
                self.tgt_deltas[drone_id].pop(0)

            # vs. obstacle avoidance
            do = min_sonar_dist
            
            # positive distance delta indicates "avoiding" an obstacle
            ndo = np.round((do - np.mean(self.obs_dists[drone_id])) / np.std(self.obs_dists[drone_id]),2)
            
            # save current values
            self.last_obs_dist[drone_id] = do
            self.obs_dists[drone_id].append(do)

            if len(self.obs_dists[drone_id]) > STATS_BUFFER:
                self.obs_dists[drone_id].pop(0)
            
            # finally, apply calcs to score move
            if cur_mode == ACQUIRE:
                acquire_move_efficiency = np.round(ndt / target_dist**0.333,2)
                # cubed root of the target distance... lessens effect of distance
            else:
                avoid_move_efficiency = np.round(ndo / target_dist**0.333,2) # was 0.333
                acquire_move_efficiency = np.round(ndt / target_dist**0.333,2)
                # for balancing avoidance with acquisition
        
        else:
            #if self.total_frame_ctr == 1 or self.replay_frame_ctr % PACK_EVAL_FRAMES == 0:
            # 3. calc normalized efficiency of last move
            # vs. target acquisition
            dt = int(self.last_pack_tgt_dist[drone_id, target_id] - target_dist)
            #print("dt:", dt)
            if abs(dt) >= 12: # mistakenly thinking crashes are moves. so, ignore moves > 12
                dt = np.mean(self.pack_tgt_deltas[drone_id])
        
            # postive distance delta indicates "closing" on the target
            ndt = np.round((dt - np.mean(self.pack_tgt_deltas[drone_id])) / np.std(self.pack_tgt_deltas[drone_id]),2)
            #print("ndt:", ndt)
            # save current values
            self.last_pack_tgt_dist[drone_id, target_id] = target_dist
            self.pack_tgt_deltas[drone_id].append(dt)
            
            if len(self.pack_tgt_deltas[drone_id]) > STATS_BUFFER:
                self.pack_tgt_deltas[drone_id].pop(0)

            # vs. obstacle avoidance
            do = min_sonar_dist
            #print("do:", do)
            # positive distance delta indicates "avoiding" an obstacle
            ndo = np.round((do - np.mean(self.pack_obs_dists[drone_id])) / np.std(self.pack_obs_dists[drone_id]),2)
            #print("ndo:", ndo)
            # save current values
            self.last_pack_obs_dist[drone_id] = do
            self.pack_obs_dists[drone_id].append(do)
        
            if len(self.pack_obs_dists[drone_id]) > STATS_BUFFER:
                self.pack_obs_dists[drone_id].pop(0)

            # finally, apply calcs to score move
            avoid_move_efficiency = np.round(ndo / target_dist**0.333,2)
            acquire_move_efficiency = np.round(ndt / target_dist**0.333,2)
            # for balancing avoidance with acquisition
            
        # 4. if w/in reasonable distance, declare victory
        if target_dist <= TARGET_RADIUS:
            print("************** target acquired ************")
            self.target_acquired = True
        
            # move acquired target to acquired targets
            self.acquired_targets.append(self.current_targets[target_id])
            self.target_inventory.remove(self.current_targets[target_id])
            
            print("pct complete:", (len(self.acquired_targets) /
                                    (len(self.acquired_targets) + len(self.target_inventory))))
                    
            if len(self.acquired_targets) % 20 == 1:
                take_screen_shot(screen)
                time.sleep(0.2) # screen capture takes a bit
            
            # remove old target
            #print("current_targets before removal:")
            #print(self.current_targets)
            self.current_targets.remove(self.current_targets[target_id])
            # get a new target
            self.assign_target(False, drone_id)

            start_dists = []
            if cur_mode == PACK:
                # find furthest two start positions
                for i in range(len(self.start_positions)):
                    dx = self.start_positions[i][0] - self.current_targets[0][0]
                    dy = self.start_positions[i][1] - self.current_targets[0][1]
                    start_dists.append(int(((dx**2 + dy**2)**0.5)))
                
                # move drones to start position
                for i in range(NUM_DRONES):
                    self.drones[i].position = \
                        self.start_positions[start_dists.index(max(min_dists))][0], \
                        self.start_positions[start_dists.index(max(min_dists))][1]
                    
                    self.drones[i].angle = self.start_angles[start_dists.index(max(start_dists))]

        #print("***** out eval_move *****")
        return target_dist, true_target_angle_rads, drone_angle_rads, adj_heading_to_target, avoid_move_efficiency, acquire_move_efficiency
            
    
    def build_states(self, drone_id, turn_action, heading_adjust, cur_speed):
        
        turn_state = 0
        avoid_state = 0
        acquire_state = 0
        hunt_state = 0
        drone_state = 0
        min_sonar_dist = 0
        avoid_move_efficiency = 0
        acquire_move_efficiency = 0
        
        # get readings from the various sensors
        sonar_dist_readings, sonar_color_readings = \
            self.get_sonar_dist_color_readings(drone_id)
        
        turn_readings = sonar_dist_readings[:TURN_NUM_SENSOR]
        min_sonar_dist = min(turn_readings)
        turn_readings = turn_readings + sonar_color_readings[:TURN_NUM_SENSOR]
        turn_state = np.array([turn_readings])
        
        if cur_mode != TURN:
            avoid_readings = sonar_dist_readings[:AVOID_NUM_SENSOR]
            min_sonar_dist = min(avoid_readings)
            avoid_readings = avoid_readings + sonar_color_readings[:AVOID_NUM_SENSOR]
            avoid_readings.append(turn_action)
            avoid_readings.append(cur_speed)
            avoid_state = np.array([avoid_readings])
        
        if cur_mode in [ACQUIRE, HUNT, PACK]:
            # calc distances, headings and efficiency
            min_sonar_dist = min(sonar_dist_readings[:HUNT_NUM_SENSOR])
            # note: avoid, hunt, pack all using 7 sensors for min dist.
            # however, pack will only be seeing four sensors. FIX THIS AT SOME POINT.
            # problem is, you can't call evaluate_move twice as it appends readings for mean sd ea time. So, some moves will be evaluated based on sensor distances it doesn't see.
            target_dist, target_angle_rads, drone_angle_rads, adj_heading_to_target, \
                avoid_move_efficiency, acquire_move_efficiency = \
                self.evaluate_move(drone_id, heading_adjust, min_sonar_dist)
        
            acquire_state = np.array([[target_dist, adj_heading_to_target]])

            if cur_mode in [HUNT, PACK]:
                hunt_readings = sonar_dist_readings[:HUNT_NUM_SENSOR]
                hunt_readings = hunt_readings + sonar_color_readings[:HUNT_NUM_SENSOR]
                hunt_readings.append(target_dist)
                hunt_readings.append(adj_heading_to_target)
                hunt_state = np.array([hunt_readings])
                min_sonar_dist = min(sonar_dist_readings[:HUNT_NUM_SENSOR])

            if cur_mode == PACK and (self.total_frame_ctr == 1 or self.replay_frame_ctr % PACK_EVAL_FRAMES == 0):
                # pack requires four compas point (above, below, right and left) obs dist readings
                compass_rads = [0, (3.14/2), 3.14, (-3.14/2)]
                drone_readings = []
                
                # it gets readings by adjusting the sonar readings for the drone angle...
                #print(drone_angle_rads)
                sonar_angles = [0, 0.6, -0.6, 1.2, -1.2, 2.8, -2.8]
                sonar_angles_adj = np.add(sonar_angles, drone_angle_rads)
                #print(sonar_angles_adj)
                
                # ...then finds the sonar reading closest to its required compass direction
                for rad in range(len(compass_rads)):
                    drone_readings.append(sonar_dist_readings[find_nearest(sonar_angles_adj,
                                                                          compass_rads[rad])])
                drone_readings.append(target_dist)
                drone_readings.append(target_angle_rads)
                drone_state = np.array([drone_readings])

        return turn_state, avoid_state, acquire_state, hunt_state, drone_state, min_sonar_dist, avoid_move_efficiency, acquire_move_efficiency


    def calc_rwd(self, drone_id, min_sonar_dist, driving_direction, cur_speed, avoid_move_efficiency, acquire_move_efficiency):

        reward = 0
        x, y = self.drones[drone_id].position

        # check for crash
        if min_sonar_dist <= 1: #  and cur_mode != PACK
            reward = -500
            if x < 0 or x > width or y < 0 or y > height:
                self.drones[drone_id].position = int(width/2), int(height/2)
                self.num_off_scrn += 1
                print("off screen. total off screens", self.num_off_scrn)
                reward = -1000
            self.recover_from_crash(driving_direction, drone_id)
        
        else:
            #print("IN ELSE")
            if cur_mode == TURN:
                # Rewards better spacing from objects
                reward = min_sonar_dist
            
            elif cur_mode == AVOID:
                # rewards distance from objects and speed
                sd_speeds = np.std(SPEEDS)
                sd_dist = np.std(range(20))
            
                std_speed = cur_speed / sd_speeds
                std_dist = min_sonar_dist / sd_dist
            
                std_max_speed = max(SPEEDS) / sd_speeds
                std_max_dist = SONAR_ARM_LEN / sd_dist
            
                reward = ((std_speed * std_dist) +
                          ((std_max_speed - std_speed) * (std_max_dist - std_dist)))
            
            else: # i.e., cur_mode is acquisition-related (acquire, hunt, pack)
                # rewards moving in the right direction and acquiring pixels
                if self.target_acquired == True:
                    reward = 1000
                    self.target_acquired = False
                    #print("frames / acquisition:", int(self.acquire_frame_ctr / NUM_DRONES))
                    self.acquire_frame_ctr = 0

                else:
                    if cur_mode == ACQUIRE:
                        reward = 100 * acquire_move_efficiency
                    
                    elif cur_mode == HUNT:
                        reward = 40 * acquire_move_efficiency + 50 * avoid_move_efficiency

        if cur_mode == PACK:
            # rewards moving all drones in right direction and acquiring pixels
            
            if reward == 1000:
                self.pack_cum_rwds[drone_id, 0] += 1000
                
            elif reward == -500 or reward == -1000:
                self.pack_cum_rwds[drone_id, 0] -= 500
        
            else:
                # two drones. reward each 1/2 of total in acquire/avoid eff proportion
                self.pack_cum_rwds[drone_id, 0] += ((30 / PACK_EVAL_FRAMES) * acquire_move_efficiency) + ((20 / PACK_EVAL_FRAMES) * avoid_move_efficiency)
            #reward = 0
            
            if self.total_frame_ctr == 1 or self.replay_frame_ctr % PACK_EVAL_FRAMES == 0:
                reward = int(self.pack_cum_rwds[drone_id, 0])
                self.pack_cum_rwds[drone_id, 0] = 0
            
            #print(reward)
            #if self.total_frame_ctr % 50000 == 0 or self.total_frame_ctr % 50000 == 1:
            #    print("***** new rcd *****")
            #    print("target dist:", target_dist)
            #    print("dt:", dt)
            #    print("mean dist deltas:", np.mean(self.tgt_deltas))
            #    print("std dist deltas:", np.std(self.tgt_deltas))
            #    print("ndt:", ndt)
            #    print("min obs dist:", min(sonar_dist_readings[:HUNT_NUM_SENSOR]))
            #    print("do:", do)
            #    print("mean obs dists:", np.mean(self.obs_dists))
            #    print("std obs dists:", np.std(self.obs_dists))
            #    print("ndo:", ndo)
            #    print("target dist ** 0.33:", target_dist**0.333)
            #    print("move eff:", move_efficiency)
            #    print("reward:", reward)
            #print("*************** 1. drone - move rewards **********")
            #print(self.total_frame_ctr)
            #print(self.pack_cum_rwds)
            #print(reward)
        return reward


    def recover_from_crash(self, driving_direction, drone_id):
        """
        we hit something, so recover
        """
        #while self.crashed:
        crash_adjust = -100
        # back up
        self.drones[drone_id].velocity = crash_adjust * driving_direction
        #self.crashed = False
        for i in range(10): # was 10
            self.drones[drone_id].angle += .2  # Turn a little.
            screen.fill(THECOLORS["red"])  # Red is scary!
            draw(screen, self.space)
            self.space.step(1./10)
            if draw_screen:
                pygame.display.flip()
            clock.tick()


    def get_sonar_dist_color_readings(self, drone_id):
        sonar_dist_readings = []
        sonar_color_readings = []
        """
        sonar readings return N "distance" readings, one for each sonar. distance is
        a count of the first non-zero color detection reading starting at the object.
        """
        
        # make sonar "arms"
        arm_1 = self.make_sonar_arm(drone_id)
        arm_2 = arm_1
        arm_3 = arm_1
        arm_4 = arm_1
        arm_5 = arm_1
        arm_6 = arm_1
        arm_7 = arm_1
        
        # rotate arms to get vector of readings
        d, c = self.get_arm_dist_color(arm_1, 0, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_2, 0.6, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_3, -0.6, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_4, 1.2, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_5, -1.2, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_6, 2.8, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        d, c = self.get_arm_dist_color(arm_7, -2.8, drone_id)
        sonar_dist_readings.append(d); sonar_color_readings.append(c)
        
        if show_sensors:
            pygame.display.update()

        return sonar_dist_readings, sonar_color_readings


    def get_arm_dist_color(self, arm, offset, drone_id):
        # count arm length to nearest obstruction
        i = 0
        x, y = self.drones[drone_id].position
        
        # evaluate each arm point to see if we've hit something
        for point in arm:
            i += 1
            
            # move the point to the right spot
            rotated_p = self.get_rotated_point(x, y, point[0], point[1],
                                               self.drones[drone_id].angle + offset)
            
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


    def make_sonar_arm(self, drone_id):
        x, y = self.drones[drone_id].position
        
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
            reading == pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR] or \
            reading == pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR] or \
            reading == pygame.color.THECOLORS[PATH_COLOR]:
            return 0
        else:
            if reading == pygame.color.THECOLORS[WALL_COLOR]:
                return 1
            elif reading == pygame.color.THECOLORS[CAT_COLOR]:
                return 2
            elif reading == pygame.color.THECOLORS[OBSTACLE_COLOR]:
                return 3
            else:
                return 1

    # ***** target and acquire model functions *****

    def generate_targets(self, first_iter):
        
        # calc number of targets that can fit space
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

    def assign_target(self, first_iter, drone_id):
        
        # clear the path surface
        path_grid.fill(pygame.color.THECOLORS[BACK_COLOR])
        
        # mark target as acquired
        x, y = self.drones[drone_id].position
        
        if first_iter == False:
            
            pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRED_PIXEL_COLOR],
                             ((x, height - y), (ACQUIRE_PIXEL_SIZE, ACQUIRE_PIXEL_SIZE)), 0)
        
        # randomly select a new target
        new_target = random.choice(self.target_inventory)
        
        #print("*** assign target ***")
        self.current_targets.append(new_target)
        #print("new target:", new_target)
        #print("current targets:", self.current_targets)
        
        # draw the new target
        pygame.draw.rect(target_grid, pygame.color.THECOLORS[ACQUIRE_PIXEL_COLOR],
                         ((new_target[0], height - new_target[1]),
                          (ACQUIRE_PIXEL_SIZE, ACQUIRE_PIXEL_SIZE)), 0)

# ***** global functions *****

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

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

