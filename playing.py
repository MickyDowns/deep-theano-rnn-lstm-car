"""
Once a model is learned, use this to play it.
"""

from flat_game import carmunk
import numpy as np
from nn import turn_net, speed_net, acquire_net, hunt_net
import random
from learning import state_frames
import time

# base operating modes based on which neural net model is training
HUNT = 1
TURN = 2
SPEED = 3
ACQUIRE = 4
cur_mode = HUNT

# record sizing settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_NUM_FRAME = 2
TURN_NUM_INPUT = TURN_NUM_FRAME * TURN_NUM_SENSOR

SPEED_NUM_SENSOR = 7
# was front five sonar distance readings, turn action, current speed
SPEED_NUM_OUTPUT = 4 # do nothing, 30, 40, 50, 60, 70
SPEED_NUM_FRAME = 3
SPEED_NUM_INPUT = SPEED_NUM_FRAME * SPEED_NUM_SENSOR
SPEEDS = [30,50,70]

ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_NUM_FRAME = 2
ACQUIRE_NUM_INPUT = ACQUIRE_NUM_FRAME * ACQUIRE_NUM_SENSOR

# hunt model settings
HUNT_NUM_SENSOR = 9 # 7 sonar sensors, distance, heading.
# could go up to 10 incl turn action, speed action and acquire action
HUNT_NUM_OUTPUT = 2 # stop, speed (model), acquire (model)
HUNT_NUM_FRAME = 3
HUNT_NUM_INPUT = HUNT_NUM_FRAME * HUNT_NUM_SENSOR

# initialize globals
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 0
START_DISTANCE = 0

def play(turn_model, turn_model_1, turn_model_2, turn_model_3,
         speed_model, acquire_model, acquire_model_1, acquire_model_2,
         hunt_model,cur_mode):

    car_distance = 0
    cum_speed = 0
    stop_ctr = speed_ctr = acquire_ctr = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    turn_state, speed_state, acquire_state, hunt_state, new_reward, cur_speed, _, _, _, _ = \
        game_state.frame_step(cur_mode, START_TURN_ACTION, START_SPEED_ACTION,
                              START_SPEED, START_DISTANCE)

    if cur_mode in [TURN, SPEED, HUNT]:
        turn_state = state_frames(turn_state,
                                  np.zeros((1, TURN_NUM_SENSOR * (TURN_NUM_FRAME - 1))),
                                  TURN_NUM_SENSOR, TURN_NUM_FRAME)
            
        if cur_mode in [SPEED, HUNT]:
            speed_state = state_frames(speed_state,
                                       np.zeros((1, SPEED_NUM_SENSOR * (SPEED_NUM_FRAME - 1))),
                                       SPEED_NUM_SENSOR, SPEED_NUM_FRAME)

    if cur_mode in [ACQUIRE, HUNT]:
        acquire_state = state_frames(acquire_state,
                                     np.zeros((1, ACQUIRE_NUM_SENSOR * (ACQUIRE_NUM_FRAME - 1))),
                                     ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)
        
    if cur_mode == HUNT:
        hunt_state = state_frames(hunt_state,
                                  np.zeros((1, HUNT_NUM_SENSOR * (HUNT_NUM_FRAME - 1))),
                                  HUNT_NUM_SENSOR, HUNT_NUM_FRAME)

    # Move.
    while True:
        car_distance += 1
        
        #time.sleep(1)
        
        # Choose action.
        speed_action = START_SPEED_ACTION

        if cur_mode == TURN:
            turn_qval = turn_model.predict(turn_state, batch_size=1)
            turn_action = (np.argmax(turn_qval))  # getting best prediction
            
        if cur_mode in [SPEED, HUNT]:
            if cur_speed == SPEEDS[0]:
                turn_qval = turn_model_1.predict(turn_state, batch_size=1)
            elif cur_speed == SPEEDS[1]:
                turn_qval = turn_model_2.predict(turn_state, batch_size=1)
            elif cur_speed in [0,SPEEDS[2]]:
                turn_qval = turn_model_3.predict(turn_state, batch_size=1)
            turn_action = (np.argmax(turn_qval))
                
            speed_state[0][5] = turn_action
            speed_qval = speed_model.predict(speed_state, batch_size=1)
            speed_action = (np.argmax(speed_qval))
            
        if cur_mode == ACQUIRE:
            acquire_qval = acquire_model.predict(acquire_state, batch_size=1)
            acquire_action = (np.argmax(acquire_qval))
            turn_action = acquire_action

        if cur_mode == HUNT:
            if cur_speed in [SPEEDS[0], SPEEDS[1]]:
                acquire_qval = acquire_model_1.predict(acquire_state, batch_size=1)
            else:
                acquire_qval = acquire_model_2.predict(acquire_state, batch_size=1)
            acquire_action = (np.argmax(acquire_qval))
                
            hunt_qval = hunt_model.predict(hunt_state, batch_size=1)
            hunt_action = (np.argmax(hunt_qval))
                
                #if hunt_action == 0: # override prior decisions and stop
                #turn_action = 0
                #speed_action = 4
                #stop_ctr += 1

            if hunt_action == 0: # accept speed model action
                turn_action = turn_action
                if cur_speed > 0:
                    speed_action = speed_action # continue current speed
                else:
                    speed_action = 2
                speed_ctr += 1
            
            elif hunt_action == 1: # accept acquire model action
                turn_action = acquire_action
                if cur_speed > 0:
                    speed_action = 0 # continue current speed
                else:
                    speed_action = 2 # reset speed to 50, as you were stopped
                    acquire_ctr += 1
                        #print("hunt_action:", hunt_action)
        # pass action, receive new state, reward
        new_turn_state, new_speed_state, new_acquire_state, new_hunt_state, new_reward, new_speed, new_rwd_turn, new_rwd_acquire, new_rwd_speed, new_rwd_hunt = game_state.frame_step(cur_mode, turn_action, speed_action, cur_speed, car_distance)

        #print(new_reward)
        # append (horizontally) historical states for learning speed.
        if cur_mode in [TURN, SPEED, HUNT]:
            new_turn_state = state_frames(new_turn_state, turn_state,
                                          TURN_NUM_SENSOR,TURN_NUM_FRAME)
        
        if cur_mode in [SPEED, HUNT]:
            new_speed_state = state_frames(new_speed_state, speed_state,
                                           SPEED_NUM_SENSOR,SPEED_NUM_FRAME)
        
        if cur_mode in [ACQUIRE, HUNT]:
            new_acquire_state = state_frames(new_acquire_state, acquire_state,
                                             ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)
        
        if cur_mode == HUNT:
            new_hunt_state = state_frames(new_hunt_state, hunt_state,
                                          HUNT_NUM_SENSOR, HUNT_NUM_FRAME)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)
            print("speed ctr:", speed_ctr, "acquire ctr:", acquire_ctr)
            stop_ctr = speed_ctr = acquire_ctr = 0

        turn_state = new_turn_state
        speed_state = new_speed_state
        acquire_state = new_acquire_state
        hunt_state = new_hunt_state
        if new_speed != cur_speed:
            print(cur_speed)
        cur_speed = new_speed
        cum_speed += cur_speed

def take_screen_shot(screen):
    time_taken = time.asctime(time.localtime(time.time()))
    time_taken = time_taken.replace(" ", "_")
    time_taken = time_taken.replace(":",".")
    save_file = "screenshots/" + time_taken + ".png"
    pygame.image.save(screen,save_file)
    print("screen shot taken")


if __name__ == "__main__":
    turn_model = turn_model_1 = turn_model_2 = turn_model_3 = speed_model = \
    acquire_model = acquire_model_1 = acquire_model_2 = hunt_model = 0

    if cur_mode == TURN:
        saved_model = 'models/turn/turn-250-100-100-50000-700000.h5'
        turn_model = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                              TURN_NUM_OUTPUT, saved_model)
    
    if cur_mode in [SPEED, HUNT]:
        saved_model = 'models/turn/turn-250-100-100-50000-30-600000.h5'
        turn_model_1 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                TURN_NUM_OUTPUT, saved_model)
        saved_model = 'models/turn/turn-250-100-100-50000-50-550000.h5'
        turn_model_2 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                TURN_NUM_OUTPUT, saved_model)
        saved_model = 'models/turn/turn-250-100-100-50000-70-450000.h5'
        turn_model_3 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                TURN_NUM_OUTPUT, saved_model)

        saved_model = 'models/speed/speed-525-105-21-100-50000-600000.h5'
        speed_model = speed_net(SPEED_NUM_INPUT,
                                [SPEED_NUM_INPUT * 25, SPEED_NUM_INPUT * 5, SPEED_NUM_INPUT],
                                SPEED_NUM_OUTPUT, saved_model)

    if cur_mode == ACQUIRE:
        saved_model = 'models/acquire/acquire-60-20-100-50000-70-400000.h5'
        acquire_model = acquire_net(ACQUIRE_NUM_INPUT, [ACQUIRE_NUM_INPUT*15,
                                                        ACQUIRE_NUM_INPUT*5],
                                    ACQUIRE_NUM_OUTPUT, saved_model)

    if cur_mode == HUNT:
        saved_model = 'models/acquire/acquire-60-20-100-50000-50-350000.h5'
        acquire_model_1 = acquire_net(ACQUIRE_NUM_INPUT, [ACQUIRE_NUM_INPUT*15,
                                                        ACQUIRE_NUM_INPUT*5],
                                    ACQUIRE_NUM_OUTPUT, saved_model)
        saved_model = 'models/acquire/acquire-60-20-100-50000-70-400000.h5'
        acquire_model_2 = acquire_net(ACQUIRE_NUM_INPUT, [ACQUIRE_NUM_INPUT*15,
                                                        ACQUIRE_NUM_INPUT*5],
                                    ACQUIRE_NUM_OUTPUT, saved_model)
                                
        saved_model = 'models/hunt/saved/hunt-675-135-27-100-50000-33-200000-best.h5'
        hunt_model = hunt_net(HUNT_NUM_INPUT,
                              [HUNT_NUM_INPUT * 25, HUNT_NUM_INPUT * 5, HUNT_NUM_INPUT],
                              HUNT_NUM_OUTPUT, saved_model)

    play(turn_model,turn_model_1, turn_model_2, turn_model_3,
         speed_model,acquire_model,acquire_model_1, acquire_model_2,
         hunt_model,cur_mode)
