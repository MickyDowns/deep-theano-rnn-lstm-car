"""
Once a model is learned, use this to play it.
"""

from flat_game import carmunk
import numpy as np
from nn import turn_net, speed_net
import random
from learning import state_frames

# operating modes based on which neural net model is training
FUTURE_STATE = 1
BEST_TURN = 2
BEST_SPEED = 3
BEST_DIST = 4
REVERSE = 5
cur_mode = BEST_TURN

if cur_mode == FUTURE_STATE:
    NUM_SENSOR = 7 # seven sonar distance readings
    NUM_OUTPUT = 7 # max one for each obstacle
elif cur_mode == BEST_TURN:
    NUM_SENSOR = 5 # front five sonar distance readings
    NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
elif cur_mode == BEST_SPEED:
    NUM_SENSOR = 7 # front five sonar distance readings, next action, current speed
    NUM_OUTPUT = 3 # do nothing, speed up, slow down
elif cur_mode == BEST_DIST:
    NUM_SENSOR = 8 # seven sonar distance readings + current speed
    NUM_OUTPUT = 2 # do nothing, reverse?
#elif cur_mode == REVERSE
#NUM_SENSOR = 6 # three front, two rear sonars, speed, next action

NUM_FRAME = 3
NUM_INPUT = NUM_FRAME * NUM_SENSOR

START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 0
START_DISTANCE = 0


def play(turn_model,speed_model):

    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    state, new_reward, cur_speed, _, _, _ = game_state.frame_step(cur_mode, START_TURN_ACTION, START_SPEED_ACTION,START_SPEED, START_DISTANCE)
    
    state = state_frames(state, np.zeros((1, NUM_SENSOR * (NUM_FRAME - 1))))

    #state, _, speed, _, _, _ = \
    #    game_state.frame_step(CUR_MODE, START_ACTION, START_SPEED, START_DISTANCE)

    # Move.
    while True:
        car_distance += 1
        
        # Choose action.
        if cur_mode == BEST_TURN or BEST_SPEED:
            turn_action = (np.argmax(turn_model.predict(state, batch_size=1)))
            speed_action = 0
        
        if cur_mode == BEST_SPEED:
            # think you're going to need to append turn_action and speed to state
            speed_action = (np.argmax(speed_model.predict(state, batch_size=1)))
    
        # Take action.
        new_state, new_reward, new_speed, new_rwd_read, new_rwd_dist, new_rwd_speed = game_state.frame_step(cur_mode, turn_action, speed_action, cur_speed, car_distance)
        
        # Append (horizontally) historical states for learning speed
        new_state = state_frames(new_state, state)
        
        #state, _, speed, _, _, _ = \
        #    game_state.frame_step(CUR_MODE, action, speed, car_distance)
        
        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

        state = new_state
        cur_speed = new_speed


if __name__ == "__main__":
    if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
        saved_model = 'models/turn/250-100-100-50000-500000.h5'
        turn_model = turn_net(NUM_INPUT, [NUM_INPUT*25, NUM_INPUT*10],
                              NUM_OUTPUT, saved_model)
        speed_model = 0
    
    if cur_mode == BEST_SPEED:
        saved_model = 'models/speed/160-150-100-50000-50000.h5'
        speed_model = speed_net(NUM_INPUT, [164, 150], NUM_OUTPUT, saved_model)

    play(turn_model,speed_model)
