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
CAR_COLOR = "blue"
CAR_SMILE = "blue"
OBSTACLE_COLOR = "purple"
CAT_COLOR = "orange"

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True # was False
draw_screen = True # was False


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
        self.create_cat()
        self.create_dog()
        self.create_goat()
        self.create_sheep()

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS[OBSTACLE_COLOR]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 35) # was 30
        self.cat_shape.color = THECOLORS[CAT_COLOR]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)
    
    def create_dog(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.dog_body = pymunk.Body(1, inertia)
        self.dog_body.position = 50, height - 600
        self.dog_shape = pymunk.Circle(self.dog_body, 35) # was 30
        self.dog_shape.color = THECOLORS[CAT_COLOR]
        self.dog_shape.elasticity = 1.0
        self.dog_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.dog_body.angle)
        self.space.add(self.dog_body, self.dog_shape)
    
    def create_goat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.goat_body = pymunk.Body(1, inertia)
        self.goat_body.position = 950, height - 100
        self.goat_shape = pymunk.Circle(self.goat_body, 35) # was 30
        self.goat_shape.color = THECOLORS[CAT_COLOR]
        self.goat_shape.elasticity = 1.0
        self.goat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.goat_body.angle)
        self.space.add(self.goat_body, self.goat_shape)
    
    def create_sheep(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.sheep_body = pymunk.Body(1, inertia)
        self.sheep_body.position = 950, height - 600
        self.sheep_shape = pymunk.Circle(self.sheep_body, 35) # was 30
        self.sheep_shape.color = THECOLORS[CAT_COLOR]
        self.sheep_shape.elasticity = 1.0
        self.sheep_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.sheep_body.angle)
        self.space.add(self.sheep_body, self.sheep_shape)

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

    def frame_step(self, action, cur_speed, car_distance):
        
        # action == 0 is continue straight
        if action == 1:  # Turn left.
            self.car_body.angle -= .2
        elif action == 2:  # Turn right.
            self.car_body.angle += .2
        elif action == 3:  # Reverse.
            self.car_body.angle += 3.14
        elif action == 4: # Slow down
            cur_speed -= 5
        elif action == 5: # Speed up
            cur_speed += 5
            
        # Move obstacles - they're 20x more stable than self
        if self.num_steps % 20 == 0: # was 100
            self.move_obstacles()

        # Move cat and dog - they're 10x more stable than self
        if self.num_steps % 5 == 0: # was 5
            self.move_cat()
            self.move_dog()
            self.move_goat()
            self.move_sheep()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = cur_speed * driving_direction # was 100
        
        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        readings.append(cur_speed)
        state = np.array([readings])

        reward_readings = round(self.sum_readings(readings)/(7*20),2)
        reward_distance = round(car_distance * 0, 2)
        reward_speed = round(cur_speed / 25, 2)

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
        else:
            # Rewards better spacing from objects, steps survived, speed of this step
            reward = round(reward_readings + reward_distance + reward_speed,2)

        # de-bug
#       readings.append(reward)
#       readings.append(reward_readings)
#        readings.append(reward_distance)
#        readings.append(reward_speed)
#        print(readings)

        self.num_steps += 1
#        print("reward:", reward)
#        print("reading:", reward_readings)
#        print("distance:", reward_distance) # rewarding cum distance at each step may not work.
#        print("speed:", reward_speed)

        return state, reward, cur_speed, reward_readings, reward_distance, reward_speed

    def move_obstacles(self):
        # Randomly move obstacles around. note: this is called every 100 frames
        for obstacle in self.obstacles:
            speed = random.randint(3, 7) # was 1 to 5
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        # this is called every 5 frames
        speed = random.randint(50, 150) # speed vary's, was 20 to 200
        self.cat_body.angle -= random.randint(-1, 1) # angle adjusts (what if we made this
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction
    
    def move_dog(self):
        # this is called every 5 frames. if you want them steadier, change to 10
        speed = random.randint(50, 150) # was 200
        self.dog_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.dog_body.angle)
        self.dog_body.velocity = speed * direction
    
    def move_goat(self):
        # this is called every 5 frames. if you want them steadier, change to 10
        speed = random.randint(50, 150) # was 200
        self.goat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.goat_body.angle)
        self.goat_body.velocity = speed * direction
    
    def move_sheep(self):
        # this is called every 5 frames. if you want them steadier, change to 10
        speed = random.randint(50, 150) # was 200
        self.sheep_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.sheep_body.angle)
        self.sheep_body.velocity = speed * direction
    
    # as long as car turns its orientation and keeps moving, it s/b able to use front 3 sensors for crash?
    def car_is_crashed(self, readings):
        # SHOULD US "ANY"
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1 or readings[3] == 1 \
            or readings[4] == 1 or readings[5] == 1 or readings[6] == 1:
            return True
        else:
            return False

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

    def get_sonar_readings(self, x, y, angle):
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
        readings.append(self.get_arm_distance(arm_1, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_2, x, y, angle, 0.9))
        readings.append(self.get_arm_distance(arm_3, x, y, angle, 1.8))
        readings.append(self.get_arm_distance(arm_4, x, y, angle, 2.7))
        readings.append(self.get_arm_distance(arm_5, x, y, angle, 3.5))
        readings.append(self.get_arm_distance(arm_6, x, y, angle, 4.4))
        readings.append(self.get_arm_distance(arm_7, x, y, angle, 5.3))
        
        #readings.append(self.get_arm_distance(arm_1, x, y, angle, 0))
        #readings.append(self.get_arm_distance(arm_2, x, y, angle, -0.90))
        #readings.append(self.get_arm_distance(arm_3, x, y, angle, 0.90))
        #readings.append(self.get_arm_distance(arm_4, x, y, angle, -1.8))
        #readings.append(self.get_arm_distance(arm_5, x, y, angle, 1.8))
        #readings.append(self.get_arm_distance(arm_6, x, y, angle, -2.7))
        #readings.append(self.get_arm_distance(arm_7, x, y, angle, 2.7))
        
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
