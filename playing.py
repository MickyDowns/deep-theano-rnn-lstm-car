"""
Once a model is learned, use this to play it.
"""

from flat_game import carmunk
import numpy as np
from nn import turn_net, speed_net, acquire_net
import random
from learning import state_frames
import time

# base operating modes based on which neural net model is training
HUNT = 1
BEST_TURN = 2
BEST_SPEED = 3
ACQUIRE = 4
cur_mode = ACQUIRE

# record sizing settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_NUM_FRAME = 2
TURN_NUM_INPUT = TURN_NUM_FRAME * TURN_NUM_SENSOR

SPEED_NUM_SENSOR = 3
# was front five sonar distance readings, turn action, current speed
SPEED_NUM_OUTPUT = 6 # do nothing, 30, 40, 50, 60, 70
SPEED_NUM_FRAME = 3
SPEED_NUM_INPUT = SPEED_NUM_FRAME * SPEED_NUM_SENSOR

ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_NUM_FRAME = 2
ACQUIRE_NUM_INPUT = ACQUIRE_NUM_FRAME * ACQUIRE_NUM_SENSOR

# initialize globals
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 0

def play(turn_model,speed_model,acquire_model,cur_mode):

    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    turn_state, speed_state, acquire_state, new_reward, cur_speed, _, _, _ = \
        game_state.frame_step(cur_mode, START_TURN_ACTION, START_SPEED_ACTION,
                              START_SPEED, START_DISTANCE)
    
    if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
        turn_state = state_frames(turn_state,
                                  np.zeros((1, TURN_NUM_SENSOR * (TURN_NUM_FRAME - 1))),
                                  TURN_NUM_SENSOR, TURN_NUM_FRAME)

    elif cur_mode == ACQUIRE:
        acquire_state = state_frames(acquire_state,
                                     np.zeros((1, ACQUIRE_NUM_SENSOR * (ACQUIRE_NUM_FRAME - 1))),
                                     ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)

    if cur_mode == BEST_SPEED:
        speed_state = state_frames(speed_state,
                                   np.zeros((1, SPEED_NUM_SENSOR * (SPEED_NUM_FRAME - 1))),
                                   SPEED_NUM_SENSOR, SPEED_NUM_FRAME)

    avg_speed = 0
    img_num = 1
    
    # Move.
    while True:
        car_distance += 1
        
        #time.sleep(1)
        
        # Choose action.
        speed_action = START_SPEED_ACTION
        
        if cur_mode in [BEST_TURN, BEST_SPEED]:
            
            turn_qval = turn_model.predict(turn_state, batch_size=1)
                
            turn_action = (np.argmax(turn_qval))
        
        elif cur_mode == ACQUIRE:
                
            acquire_qval = acquire_model.predict(acquire_state, batch_size=1)
                
            turn_action = (np.argmax(acquire_qval))
            
        if cur_mode == BEST_SPEED:
                
            speed_qval = speed_model.predict(speed_state, batch_size=1)
                
            speed_action = (np.argmax(speed_qval))

        # Take action, observe new state and get reward
        new_turn_state, new_speed_state, new_acquire_state, new_reward, new_speed, new_rwd_read, new_rwd_dist, new_rwd_speed = game_state.frame_step(cur_mode, turn_action, speed_action, cur_speed, car_distance)
    
        # Append (horizontally) historical states for learning speed
        if cur_mode in [BEST_TURN, BEST_SPEED]:
            new_turn_state = state_frames(new_turn_state, turn_state,
                                          TURN_NUM_SENSOR,TURN_NUM_FRAME)
        
        elif cur_mode == ACQUIRE:
            new_acquire_state = state_frames(new_acquire_state, acquire_state,
                                             ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)

        if cur_mode == BEST_SPEED:
            new_speed_state = state_frames(new_speed_state, speed_state,
                                           SPEED_NUM_SENSOR,SPEED_NUM_FRAME)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)
            print("Average speed: %d." % avg_speed)
            avg_speed = 0

        turn_state = new_turn_state
        speed_state = new_speed_state
        acquire_state = new_acquire_state
        cur_speed = new_speed


def take_screen_shot(screen):
    time_taken = time.asctime(time.localtime(time.time()))
    time_taken = time_taken.replace(" ", "_")
    time_taken = time_taken.replace(":",".")
    save_file = "screenshots/" + time_taken + ".png"
    pygame.image.save(screen,save_file)
    print("screen shot taken")


if __name__ == "__main__":
    if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
        saved_model = 'models/turn/turn-250-100-100-50000-700000.h5'
        turn_model = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                              TURN_NUM_OUTPUT, saved_model)
        speed_model = 0
        acquire_model = 0
    
    elif cur_mode == ACQUIRE:
        saved_model = 'models/acquire/acquire-60-20-100-50000-350000.h5'
        acquire_model = acquire_net(ACQUIRE_NUM_INPUT, [ACQUIRE_NUM_INPUT*15,
                                                        ACQUIRE_NUM_INPUT*5],
                                    ACQUIRE_NUM_OUTPUT, saved_model)
        turn_model = 0
        speed_model = 0
    
    if cur_mode == BEST_SPEED:
        saved_model = 'models/speed/speed-525-210-100-50000-700000.h5'
        speed_model = speed_net(SPEED_NUM_INPUT, [SPEED_NUM_INPUT*25, SPEED_NUM_INPUT*10], SPEED_NUM_OUTPUT, saved_model)

    play(turn_model,speed_model,acquire_model,cur_mode)
