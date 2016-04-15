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

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
CAR_COLOR = "green"
CAR_SMILE = "blue"
OBSTACLE_COLOR = "purple"
CAT_COLOR = "orange"

# operating modes based on which neural net model is training
FUTURE_STATE = 1
BEST_TURN = 2
BEST_SPEED = 3
BEST_DIST = 4
REVERSE = 5

# operating parameters
MIN_SPEED = 20
MAX_SPEED = 100
AVG_READING = 1
MIN_READING = 2
reading_reward_basis = MIN_READING

# training parameters
FUTURE_FLIPS = 1

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = False # was False
draw_screen = False # was False


class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(width/2, height/2, 0.5)

        # Record steps.
        self.num_steps = 0
        self.num_off_scrn = 0
        #self.xy_history = [(0,0)]

        # Create walls.
        static = [pymunk.Segment(self.space.static_body,(0, 1), (0, height), 1),
                  pymunk.Segment(self.space.static_body,(1, height), (width, height), 1),
                  pymunk.Segment(self.space.static_body,(width-1, height), (width-1, 1), 1),
                  pymunk.Segment(self.space.static_body,(1, 1), (width, 1), 1)]
        
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly. They'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100)) # was 100
        self.obstacles.append(self.create_obstacle(700, 200, 125)) # was 125
        self.obstacles.append(self.create_obstacle(600, 600, 50)) # was 35
        
        # Create a cat and dog.
        self.cats = []
        self.cats.append(self.create_cat(50,height-100))
        self.cats.append(self.create_cat(50,height-600))
        self.cats.append(self.create_cat(950,height-100))
        self.cats.append(self.create_cat(950,height-600))

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
        cat_shape = pymunk.Circle(cat_body, 35) # was 30
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
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS[CAR_COLOR]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, cur_mode, turn_action, speed_action, cur_speed, car_distance):
 
        if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
            # action == 0 is continue straight
            if turn_action == 1:  # slight left.
                self.car_body.angle -= .2
            elif turn_action == 2:  # hard left.
                self.car_body.angle -= .4
            elif turn_action == 3:  # slight right.
                self.car_body.angle += .2
            elif turn_action == 4:  # hard right.
                self.car_body.angle += .4
        
        if cur_mode == BEST_SPEED:
            # action == 0 is no change
            if speed_action == 1: # Slow downs
                if cur_speed > MIN_SPEED:
                    cur_speed -= 2
            elif speed_action == 2: # Speed up
                if cur_speed < MAX_SPEED:
                    cur_speed += 2
            #elif turn_action == 3:  # Reverse.
                #self.car_body.angle += 3.14
        
        ####for flip in range(FUTURE_FLIPS):
        
        # Move obstacles - they're 20x more stable than self
        if self.num_steps % 20 == 0: # was 100
            self.move_obstacles()

        # Move cat and dog - they're 5x more stable than self
        if self.num_steps % 5 == 0:
            self.move_cats()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = cur_speed * driving_direction # was 100

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle, cur_mode)
        
        if cur_mode == BEST_SPEED:
            readings.append(turn_action)
            readings.append(cur_speed)
        state = np.array([readings])
        
        # Calculate rewards based on training mode
        if cur_mode == BEST_TURN:
            if reading_reward_basis == AVG_READING:
                reward_readings = int(self.sum_readings(readings)/5)
            elif reading_reward_basis == MIN_READING:
                reward_readings = min(readings)
            
            reward_new_xy = 0
            reward_speed = 0 #round(abs(cur_speed) / 25, 2)
            #self.xy_history.append((rnd_x,rnd_y))
        elif cur_mode == BEST_SPEED:
            reward_readings = 0
            reward_new_xy = 0
            reward_speed = cur_speed #round(abs(cur_speed) / 25, 2)

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            if x < 0 or x > width or y < 0 or y > height:
                self.car_body.position = int(width/2), int(height/2)
                x, y = self.car_body.position
                self.num_off_scrn += 1
                print("off screen. total off screens", self.num_off_scrn)
                reward = -1000
            self.recover_from_crash(driving_direction)
            #self.xy_history = [(0,0)]
        else:
            # Rewards better spacing from objects, steps survived, speed of this step
            reward = reward_readings + reward_new_xy + reward_speed

        #### if flip == 0 and FUTURE_FLIPS > 1:
        ####    save_state() i.e., obstacles(x,y, velocity), cats(), cars(), reward
        #### then, run thru iters saving rewards
        #### exit the loop, restore_state(), keep worst reward

        self.num_steps += 1
        clock.tick()

        return state, reward, cur_speed, reward_readings, reward_new_xy, reward_speed

    def move_obstacles(self):
        # Randomly move obstacles around. note: this is called every 100 frames
        for obstacle in self.obstacles:
            speed = random.randint(3, 7) # was 1 to 5
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cats(self):
        # this is called every 5 frames
        for cat in self.cats:
            speed = random.randint(50, 150) # speed vary's, was 20 to 200
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            cat.velocity = speed * direction
    
    def car_is_crashed(self, readings):
        return_val = False
        
        for reading in readings:
            if reading == 1:
                return_val = True
        return return_val

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
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

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle, cur_mode):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        
        # Make our arms.
        arm_1 = self.make_sonar_arm(x, y)
        arm_2 = arm_1
        arm_3 = arm_1
        arm_4 = arm_1
        arm_5 = arm_1
        arm_6 = arm_1
        arm_7 = arm_1
        
        # Rotate them and get readings.
        if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
            readings.append(self.get_arm_distance(arm_1, x, y, angle, 0))
            readings.append(self.get_arm_distance(arm_2, x, y, angle, 0.6))
            readings.append(self.get_arm_distance(arm_3, x, y, angle, -0.6))
            readings.append(self.get_arm_distance(arm_4, x, y, angle, 1.2))
            readings.append(self.get_arm_distance(arm_5, x, y, angle, -1.2))
        
        if cur_mode == REVERSE:
            readings.append(self.get_arm_distance(arm_1, x, y, angle, 0))
            readings.append(self.get_arm_distance(arm_2, x, y, angle, 0.9))
            readings.append(self.get_arm_distance(arm_3, x, y, angle, -0.9))
            readings.append(self.get_arm_distance(arm_6, x, y, angle, 2.7))
            readings.append(self.get_arm_distance(arm_7, x, y, angle, -2.7))
        
        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1
            
            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )
            
            # Return i if rotated point is off board
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                # this gets the color of the pixel at the rotated point
                if self.get_track_or_not(obs) != 0:
                    # if pixel not black or green return i
                    return i

            # plots the individual sonar point on the screen
            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor. WAS 20
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((x + distance + (spread * i), y))

        return arm_points

    # SOMETHING ABOUT THIS CALC IS SLIGHTLY OFF AND CONTINUES TO GET WORSE
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
        # checking to see if the color returned corresponds to the backgroun or car
        if reading == THECOLORS['black'] or reading == THECOLORS[CAR_COLOR] or reading == THECOLORS[CAR_SMILE]:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
