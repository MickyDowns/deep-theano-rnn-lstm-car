"""
Once a model is learned, use this to play it.
"""

from flat_game import carmunk
import numpy as np
from nn import turn_net, avoid_net, acquire_net, hunt_net, pack_net
import random
from learning import state_frames
import time

# base operating modes based on which neural net model is training
TURN = 1
AVOID = 2
ACQUIRE = 3
HUNT = 4
PACK = 5
cur_mode = PACK

# record sizing settings
TURN_TOTAL_SENSORS = 10 # front five sonar color / distance readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_STATE_FRAMES = 3
TURN_NUM_INPUT = TURN_STATE_FRAMES * TURN_TOTAL_SENSORS

# avoid model settings
AVOID_TOTAL_SENSORS = 16 # seven sonar distance readings, turn action, current speed
AVOID_NUM_OUTPUT = 3 # do nothing, stop, 30, 50, 70
AVOID_STATE_FRAMES = 3
AVOID_NUM_INPUT = AVOID_STATE_FRAMES * AVOID_TOTAL_SENSORS
SPEEDS = [30,50,70]

ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_STATE_FRAMES = 2
ACQUIRE_NUM_INPUT = ACQUIRE_STATE_FRAMES * ACQUIRE_NUM_SENSOR

# hunt model settings
HUNT_AVOID = 0
HUNT_ACQUIRE = 1
HUNT_TOTAL_SENSORS = 16 # 7 sonar sensors, distance, heading.
# could go up to 10 incl turn action, speed action and acquire action
HUNT_NUM_OUTPUT = 2 # stop, speed (model), acquire (model)
HUNT_STATE_FRAMES = 3
HUNT_NUM_INPUT = HUNT_STATE_FRAMES * HUNT_TOTAL_SENSORS

# pack model settings
NUM_DRONES = 2
DRONE_TOTAL_SENSOR = 6 # obstacle (compass) distances, true target heading, target distance
PACK_TOTAL_SENSORS = DRONE_TOTAL_SENSOR * NUM_DRONES
# these are radian adjustments to first (lhs) and second (rhs) to HTT. pos is left, neg in right.
PACK_NUM_OUTPUT = 9
PACK_STATE_FRAMES = 3
PACK_EVAL_FRAMES = 5
PACK_NUM_INPUT = PACK_STATE_FRAMES * PACK_TOTAL_SENSORS
START_PACK_ACTION = 0
START_DRONE_ID = 0

# initialize globals
use_existing_model = True
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 1
START_DISTANCE = 1

def play(turn_model, turn_model_30, turn_model_50, turn_model_70, avoid_model, acquire_model,
         acquire_model_30, acquire_model_50, acquire_model_70, hunt_model, pack_model, params):

    total_frame_ctr = 0
    crash_frame_ctr = 0
    replay_frame_ctr = 0
    crash_ctr = 0
    acquire_ctr = 0
    cum_speed = 0
    stop_ctr = avoid_ctr = acquire_ctr = 0
    cur_speeds = []
    for i in range(NUM_DRONES): cur_speeds.append(START_SPEED)
    
    # initialize drone state holders
    turn_states = np.zeros([NUM_DRONES, TURN_TOTAL_SENSORS * TURN_STATE_FRAMES])
    avoid_states = np.zeros([NUM_DRONES, AVOID_TOTAL_SENSORS * AVOID_STATE_FRAMES])
    acquire_states = np.zeros([NUM_DRONES, ACQUIRE_NUM_SENSOR * ACQUIRE_STATE_FRAMES])
    hunt_states = np.zeros([NUM_DRONES, HUNT_TOTAL_SENSORS * HUNT_STATE_FRAMES])
    drone_states = np.zeros([NUM_DRONES, DRONE_TOTAL_SENSOR * PACK_STATE_FRAMES])

    # create game instance
    game_state = carmunk.GameState()

    # get initial state(s)
    turn_state, avoid_state, acquire_state, hunt_state, drone_state, reward, cur_speed = \
        game_state.frame_step(START_DRONE_ID, START_TURN_ACTION, START_SPEED_ACTION,
                              START_PACK_ACTION, START_SPEED, START_DISTANCE, 1)
    
    # initialize frame states
    if cur_mode in [TURN, AVOID, HUNT, PACK]:
        
        for i in range(NUM_DRONES): turn_states[i] = state_frames(turn_state, np.zeros((1, TURN_TOTAL_SENSORS * TURN_STATE_FRAMES)), TURN_TOTAL_SENSORS, TURN_STATE_FRAMES)
        
        if cur_mode in [AVOID, HUNT, PACK]:
            
            for i in range(NUM_DRONES): avoid_states[i] = state_frames(avoid_state, np.zeros((1, AVOID_TOTAL_SENSORS * AVOID_STATE_FRAMES)),AVOID_TOTAL_SENSORS, AVOID_STATE_FRAMES)

    if cur_mode in [ACQUIRE, HUNT, PACK]:
    
        for i in range(NUM_DRONES): acquire_states[i] = state_frames(acquire_state, np.zeros((1, ACQUIRE_NUM_SENSOR * ACQUIRE_STATE_FRAMES)), ACQUIRE_NUM_SENSOR, ACQUIRE_STATE_FRAMES)
    
    if cur_mode in [HUNT, PACK]:
        
        for i in range(NUM_DRONES): hunt_states[i] = state_frames(hunt_state, np.zeros((1, HUNT_TOTAL_SENSORS * HUNT_STATE_FRAMES)), HUNT_TOTAL_SENSORS, HUNT_STATE_FRAMES)
    
    if cur_mode == PACK:
        
        for i in range(NUM_DRONES): drone_states[i] = state_frames(drone_state, np.zeros((1, DRONE_TOTAL_SENSOR * PACK_STATE_FRAMES)),DRONE_TOTAL_SENSOR, PACK_STATE_FRAMES)
        
        #pack_state = state_frames(drone_state,
        #                          np.zeros((1, PACK_TOTAL_SENSORS * PACK_STATE_FRAMES)),
        #                          PACK_TOTAL_SENSORS, PACK_STATE_FRAMES)
                                  
        pack_state = state_frames(drone_state, np.zeros((1, 30)),10, 4)
    
    # Move.
    while True:
        
        total_frame_ctr += 1
        crash_frame_ctr += 1
        replay_frame_ctr += 1
        
        #time.sleep(1)
        
        for drone_id in range(NUM_DRONES): # NUM_DRONES = 1, unless you're in PACK mode
        
            speed_action = START_SPEED_ACTION
            
            # choose action
            if cur_mode == TURN:
                turn_action, turn_model = set_turn_action(False, cur_speeds[drone_id], np.array([turn_states[drone_id]]))
            else:
                if cur_mode in [AVOID, HUNT, PACK]:
                    turn_action, turn_model = set_turn_action(False, cur_speeds[drone_id],np.array([turn_states[drone_id]]))
                                                                      
                if cur_mode == AVOID:
                    speed_action = set_avoid_action(False, turn_action, np.array([avoid_states[drone_id]]))
                else:
                    if cur_mode in [HUNT, PACK]:
                        speed_action = set_avoid_action(False, turn_action, np.array([avoid_states[drone_id]]))
                                                                                          
                    if cur_mode == ACQUIRE:
                        acquire_action, acquire_model = set_acquire_action(False, cur_speeds[drone_id], np.array([acquire_states[drone_id,]]))
                        turn_action = acquire_action
                    else:
                        acquire_action, acquire_model = set_acquire_action(False, cur_speeds[drone_id], np.array([acquire_states[drone_id,]]))
                                                                                                              
                        if cur_mode == HUNT:
                            hunt_action, turn_action, speed_action = set_hunt_action(False, cur_speeds[drone_id], turn_action, speed_action, acquire_action, np.array([hunt_states[drone_id,]]))
                        else:
                            hunt_action, turn_action, speed_action = set_hunt_action(False, cur_speeds[drone_id], turn_action, speed_action, acquire_action, np.array([hunt_states[drone_id,]]))
                            
                            if cur_mode == PACK and (total_frame_ctr == 1 or replay_frame_ctr % PACK_EVAL_FRAMES == 0) and drone_id == 0:
                                # get 1 pack action for each set of drones on first drone
                                pack_action = set_pack_action(False, pack_state)

            # pass action, receive new state, reward
            new_turn_state, new_avoid_state, new_acquire_state, new_hunt_state, new_drone_state, new_reward, new_speed = game_state.frame_step(drone_id, turn_action, speed_action, pack_action, cur_speeds[drone_id], total_frame_ctr, replay_frame_ctr)

            # append (horizontally) historical states for learning speed.
            if cur_mode in [TURN, AVOID, HUNT, PACK]:
                new_turn_state = state_frames(new_turn_state,
                                              np.array([turn_states[drone_id]]),
                                              TURN_TOTAL_SENSORS,TURN_STATE_FRAMES)
            
            if cur_mode in [AVOID, HUNT, PACK]:
                new_avoid_state = state_frames(new_avoid_state,
                                               np.array([avoid_states[drone_id]]),
                                               AVOID_TOTAL_SENSORS, AVOID_STATE_FRAMES)
            
            if cur_mode in [ACQUIRE, HUNT, PACK]:
                new_acquire_state = state_frames(new_acquire_state,
                                                 np.array([acquire_states[drone_id]]),
                                                 ACQUIRE_NUM_SENSOR, ACQUIRE_STATE_FRAMES)
            
            if cur_mode in [HUNT, PACK]:
                new_hunt_state = state_frames(new_hunt_state,
                                              np.array([hunt_states[drone_id]]),
                                              HUNT_TOTAL_SENSORS, HUNT_STATE_FRAMES)
                    
            if cur_mode == PACK and (total_frame_ctr == 1 or replay_frame_ctr % PACK_EVAL_FRAMES == 0):
                if drone_id == 0: # for 1st drone, pack state = drone state
                    new_pack_state = new_drone_state
                    pack_rwd = new_reward
                        
                else: # otherwise, append drone record to prior drone state
                    new_pack_state = state_frames(new_pack_state, new_drone_state,
                                                  DRONE_TOTAL_SENSOR, PACK_STATE_FRAMES - 1)
                    pack_rwd += new_reward
                        
                new_drone_state = state_frames(new_drone_state,
                                               np.array([drone_states[drone_id]]),
                                               DRONE_TOTAL_SENSOR, PACK_STATE_FRAMES)
            
                if drone_id == (NUM_DRONES - 1): # for last drone build pack record
                    if total_frame_ctr == 1:
                        pack_state = np.zeros((1, PACK_TOTAL_SENSORS * PACK_STATE_FRAMES))
                    
                    new_pack_state = state_frames(new_pack_state, pack_state, PACK_TOTAL_SENSORS, PACK_STATE_FRAMES) #may need to add 1 to PACK_STATE_FRAMES
                        
            # Update the starting state with S'.
            if cur_mode in [TURN, AVOID, HUNT, PACK]:
                turn_states[drone_id] = new_turn_state
            
            if cur_mode in [AVOID, HUNT, PACK]:
                avoid_states[drone_id] = new_avoid_state
            
            if cur_mode in [ACQUIRE, HUNT, PACK]:
                acquire_states[drone_id] = new_acquire_state
            
            if cur_mode in [HUNT, PACK]:
                hunt_states[drone_id] = new_hunt_state
            
            if cur_mode == PACK and (total_frame_ctr == 1 or replay_frame_ctr % PACK_EVAL_FRAMES == 0):
                drone_states[drone_id] = new_drone_state
                
                if drone_id == (NUM_DRONES - 1):
                    pack_state = new_pack_state
                    replay_frame_ctr = 0
        
            cur_speeds[drone_id] = new_speed
        
        # give status
        if new_reward == -500 or new_reward == -1000:
            crash_ctr += 1
            print("crashes", crash_ctr, "frames", total_frame_ctr)
        elif new_reward == 1000:
            acquire_ctr += 1
            print("acquisitions:", acquire_ctr, "frames", total_frame_ctr)
        
        if total_frame_ctr % 5000 == 0:
            print("***** total frames:", total_frame_ctr)
            print("***** frames between crashes:", int(total_frame_ctr / crash_ctr))
                  
            if cur_mode in [ACQUIRE, HUNT, PACK]:
                  
                print("***** frames / acquisition:", int(total_frame_ctr / acquire_ctr))
            #stop_ctr = avoid_ctr = acquire_ctr = 0


def set_turn_action(random_fl, cur_speed, turn_state):
    if random_fl:
        turn_action = np.random.randint(0, TURN_NUM_OUTPUT)
        return turn_action
    else:
        if cur_mode == TURN and use_existing_model == False:
            turn_qval = turn_model.predict(turn_state, batch_size=1)
            turn_model = turn_model
        else:
            if cur_speed == SPEEDS[0]:
                turn_qval = turn_model_30.predict(turn_state, batch_size=1)
                turn_model = turn_model_30
            elif cur_speed == SPEEDS[1]:
                turn_qval = turn_model_50.predict(turn_state, batch_size=1)
                turn_model = turn_model_50
            elif cur_speed == SPEEDS[2]:
                turn_qval = turn_model_70.predict(turn_state, batch_size=1)
                turn_model = turn_model_70
        turn_action = (np.argmax(turn_qval))
        return turn_action, turn_model


def set_avoid_action(random_fl, turn_action, avoid_state):
    if random_fl:
        speed_action = np.random.randint(0, AVOID_NUM_OUTPUT)
    else:
        avoid_state[0][14] = turn_action # ensures AVOID using current turn pred
        avoid_qval = avoid_model.predict(avoid_state, batch_size=1)
        speed_action = (np.argmax(avoid_qval))
    
    return speed_action


def set_acquire_action(random_fl, cur_speed, acquire_state):
    if random_fl:
        turn_action = np.random.randint(0, ACQUIRE_NUM_OUTPUT)
        return turn_action
    else:
        if cur_mode == ACQUIRE and use_existing_model == False:
            acquire_qval = acquire_model.predict(acquire_state, batch_size=1)
            acquire_model = acquire_model
        else:
            if cur_speed == SPEEDS[0]:
                acquire_qval = acquire_model_50.predict(acquire_state, batch_size=1)
                acquire_model = acquire_model_50
            
            elif cur_speed == SPEEDS[1]:
                acquire_qval = acquire_model_50.predict(acquire_state, batch_size=1)
                acquire_model = acquire_model_50
            
            else:
                acquire_qval = acquire_model_70.predict(acquire_state, batch_size=1)
                acquire_model = acquire_model_70

    turn_action = (np.argmax(acquire_qval))
    return turn_action, acquire_model


def set_hunt_action(random_fl, cur_speed, turn_action, speed_action, acquire_action, hunt_state):
    if random_fl:
        hunt_action = np.random.randint(0, HUNT_NUM_OUTPUT)
        if hunt_action == HUNT_AVOID: # accept speed model action
            turn_action = turn_action
            if cur_speed > 0:
                speed_action = speed_action # continue current speed
            else:
                speed_action = 1 # reset speed to 50, as you were stopped
            #avoid_ctr += 1
        
        elif hunt_action == HUNT_ACQUIRE: # accept acquire model action
            turn_action = acquire_action
            if cur_speed > 0:
                speed_action = 1 # just setting acquire speed to 50 for now
            else:
                speed_action = 1 # reset speed to 50, as you were stopped
            #acquire_ctr += 1
    else:
        hunt_qval = hunt_model.predict(hunt_state, batch_size=1)
        hunt_action = (np.argmax(hunt_qval))
        
        if hunt_action == HUNT_AVOID: # accept avoid model action
            turn_action = turn_action
            if cur_speed > 0:
                speed_action = speed_action # continue current speed
            else:
                speed_action = 1
            #avoid_ctr += 1

        elif hunt_action == HUNT_ACQUIRE: # accept acquire model action
            turn_action = acquire_action
            if cur_speed > 0:
                speed_action = 1 # just setting acquire speed to 50 for now
            else:
                speed_action = 1 # reset acquire speed to 50, as you were stopped
            #acquire_ctr += 1
    
    return hunt_action, turn_action, speed_action


def set_pack_action(random_fl, pack_state):
    if random_fl:
        pack_action = np.random.randint(0, PACK_NUM_OUTPUT)
    else:
        pack_qval = pack_model.predict(pack_state, batch_size=1)
        pack_action = (np.argmax(pack_qval))
    
    return pack_action


def take_screen_shot(screen):
    time_taken = time.asctime(time.localtime(time.time()))
    time_taken = time_taken.replace(" ", "_")
    time_taken = time_taken.replace(":",".")
    save_file = "screenshots/" + time_taken + ".png"
    pygame.image.save(screen,save_file)
    print("screen shot taken")


if __name__ == "__main__":
    turn_model = turn_model_30 = turn_model_50 = turn_model_70 = avoid_model = \
        acquire_model = acquire_model_30 = acquire_model_50 = acquire_model_70 = \
        hunt_model = pack_model = 0
        
    if cur_mode in [TURN, AVOID, HUNT, PACK]:
        nn_param = [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10]
        # 3-layer [25x, 5x, x] performed horribly w/ and w/out color
        params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}
            
        if cur_mode == TURN and use_existing_model == False:
            turn_model = turn_net(TURN_NUM_INPUT, nn_param, TURN_NUM_OUTPUT)
        
        else:
            saved_model = 'models/turn/saved/turn-750-300-100-50000-30-600000.h5'
            turn_model_30 = turn_net(TURN_NUM_INPUT, nn_param, TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-50-600000.h5'
            turn_model_50 = turn_net(TURN_NUM_INPUT, nn_param, TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-70-600000.h5'
            turn_model_70 = turn_net(TURN_NUM_INPUT, nn_param,TURN_NUM_OUTPUT, saved_model)
    
    if cur_mode in [AVOID, HUNT, PACK]:
        nn_param = [AVOID_NUM_INPUT * 25, AVOID_NUM_INPUT * 5, AVOID_NUM_INPUT]
        params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}
            
        if cur_mode == AVOID and use_existing_model == False:
            avoid_model = avoid_net(AVOID_NUM_INPUT, nn_param, AVOID_NUM_OUTPUT)
            
        else:
            saved_model = 'models/avoid/saved/avoid-1200-240-48-100-50000-700000-old-3L-2022.h5'
            avoid_model = avoid_net(AVOID_NUM_INPUT, nn_param, AVOID_NUM_OUTPUT, saved_model)

    if cur_mode in [ACQUIRE, HUNT, PACK]:
        nn_param = [ACQUIRE_NUM_INPUT * 15, ACQUIRE_NUM_INPUT * 5]
        params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}
            
        if cur_mode == ACQUIRE and use_existing_model == False:
            acquire_model = acquire_net(ACQUIRE_NUM_INPUT, nn_param, ACQUIRE_NUM_OUTPUT)
        
        else:
            saved_model = 'models/acquire/saved/acquire-60-20-100-50000-50-350000.h5'
            # using 50 until time to re-train at 30
            acquire_model_30 = acquire_net(ACQUIRE_NUM_INPUT, nn_param,
                                           ACQUIRE_NUM_OUTPUT, saved_model)
            saved_model = 'models/acquire/saved/acquire-60-20-100-50000-50-350000.h5'
            acquire_model_50 = acquire_net(ACQUIRE_NUM_INPUT, nn_param,
                                           ACQUIRE_NUM_OUTPUT, saved_model)
            saved_model = 'models/acquire/saved/acquire-60-20-100-50000-70-350000.h5'
            acquire_model_70 = acquire_net(ACQUIRE_NUM_INPUT, nn_param,
                                           ACQUIRE_NUM_OUTPUT, saved_model)
        
    if cur_mode in [HUNT, PACK]:
        nn_param = [HUNT_NUM_INPUT * 25, HUNT_NUM_INPUT * 5, HUNT_NUM_INPUT]
        params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}
            
        if cur_mode == HUNT and use_existing_model == False:
            hunt_model = hunt_net(HUNT_NUM_INPUT, nn_param, HUNT_NUM_OUTPUT)
            
        else:
            saved_model = 'models/hunt/saved/hunt-1200-240-48-100-50000-300000-40-50-avoid.h5'
            hunt_model = hunt_net(HUNT_NUM_INPUT, nn_param, HUNT_NUM_OUTPUT, saved_model)
        
    if cur_mode == PACK:
        nn_param = [PACK_NUM_INPUT * 10]
        params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}
            
        if cur_mode == PACK and use_existing_model == False:
            pack_model = pack_net(PACK_NUM_INPUT, nn_param, PACK_NUM_OUTPUT)
    
        else:
            saved_model = 'models/pack/pack-360-100-50000-200000.h5'
            pack_model = pack_net(PACK_NUM_INPUT, nn_param, PACK_NUM_OUTPUT, saved_model)

    play(turn_model, turn_model_30, turn_model_50, turn_model_70, avoid_model,
          acquire_model, acquire_model_30, acquire_model_50, acquire_model_70,
          hunt_model, pack_model, params)