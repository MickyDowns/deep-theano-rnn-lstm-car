from flat_game import carmunk
import numpy as np
import random
import csv
from nn import turn_net, avoid_net, acquire_net, hunt_net, pack_net, LossHistory
import os.path
import timeit
import time

# operating modes drive neural net training
TURN = 1
AVOID = 2
ACQUIRE = 3
HUNT = 4
PACK = 5
cur_mode = TURN

# turn model settings
TURN_NUM_SENSOR = 10 # front five sonar distance and color readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_STATE_FRAMES = 3
TURN_NUM_INPUT = TURN_STATE_FRAMES * TURN_NUM_SENSOR

# speed model settings
AVOID_NUM_SENSOR = 16 # seven sonar distance + color readings, turn action, current speed
AVOID_NUM_OUTPUT = 3 # 30, 50, 70
AVOID_STATE_FRAMES = 3
AVOID_NUM_INPUT = AVOID_STATE_FRAMES * AVOID_NUM_SENSOR
SPEEDS = [30,50,70]

# acquire model settings
ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_STATE_FRAMES = 2
ACQUIRE_NUM_INPUT = ACQUIRE_STATE_FRAMES * ACQUIRE_NUM_SENSOR

# hunt model settings
HUNT_AVOID = 0
HUNT_ACQUIRE = 1
HUNT_NUM_SENSOR = 16 # seven sonar distance + color, target distance + heading
HUNT_NUM_OUTPUT = 2 # avoid (model), acquire (model)
# try w/ stop option
HUNT_STATE_FRAMES = 3
HUNT_NUM_INPUT = HUNT_STATE_FRAMES * HUNT_NUM_SENSOR

# pack model settings
NUM_DRONES = 1
DRONE_NUM_SENSOR = 6 # obstacle (compass) distances, true target heading, target distance
PACK_NUM_SENSOR = DRONE_NUM_SENSOR * NUM_DRONES
# these are radian adjustments to first (lhs) and second (rhs) to HTT. pos is left, neg in right.
PACK_NUM_OUTPUT = 9
PACK_STATE_FRAMES = 3
PACK_EVAL_FRAMES = 7
PACK_NUM_INPUT = PACK_STATE_FRAMES * PACK_NUM_SENSOR
START_PACK_ACTION = 0
START_DRONE_ID = 0

# initial settings
use_existing_model = False
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 1
START_DISTANCE = 1

# misc
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.

def train_net(turn_model, turn_model_30, turn_model_50, turn_model_70, avoid_model, acquire_model,
              acquire_model_30, acquire_model_50, acquire_model_70, hunt_model, pack_model, params):
    
    filename = params_to_filename(params)
    
    if cur_mode in [TURN, HUNT, PACK]:
        observe = 1000  # Number of frames to observe before training.
    else:
        observe = 2000

    epsilon = 1 # vary this based on pre-learning already occurred in lower models
    train_frames = 1000000  # number of flips for training
    batchSize = params['batchSize']
    buffer = params['buffer']

    # initialize variables and structures used below.
    max_crash_frame_ctr = crash_frame_ctr = total_frame_ctr = 0
    stop_ctr = avoid_ctr = acquire_ctr = cum_rwd = cum_speed = 0

    data_collect = replay = loss_log = [] # replay stores state, action, reward, new state
    save_init = True
    
    # initialize drone state holders
    turn_states = np.zeros([NUM_DRONES, TURN_NUM_SENSOR * TURN_STATE_FRAMES])
    avoid_states = np.zeros([NUM_DRONES, AVOID_NUM_SENSOR * AVOID_STATE_FRAMES])
    acquire_states = np.zeros([NUM_DRONES, ACQUIRE_NUM_SENSOR * ACQUIRE_STATE_FRAMES])
    hunt_states = np.zeros([NUM_DRONES, HUNT_NUM_SENSOR * HUNT_STATE_FRAMES])
    drone_states = np.zeros([NUM_DRONES, DRONE_NUM_SENSOR * PACK_STATE_FRAMES])
    
    # create game instance
    game_state = carmunk.GameState()
    
    # get initial state(s)
    turn_state, avoid_state, acquire_state, hunt_state, drone_state, reward, cur_speed = \
        game_state.frame_step(START_DRONE_ID, START_TURN_ACTION, START_SPEED_ACTION,
                              START_PACK_ACTION, START_SPEED, START_DISTANCE)

    # these were all frames-1
    if cur_mode in [TURN, AVOID, HUNT, PACK]:
        
        for i in range(NUM_DRONES): turn_states[i] = state_frames(turn_state,np.zeros((1, TURN_NUM_SENSOR * TURN_STATE_FRAMES)), TURN_NUM_SENSOR, TURN_STATE_FRAMES)
        
        if cur_mode in [AVOID, HUNT, PACK]:
            for i in range(NUM_DRONES): avoid_states[i] = state_frames(avoid_state, np.zeros((1, AVOID_NUM_SENSOR * AVOID_STATE_FRAMES)),AVOID_NUM_SENSOR, AVOID_STATE_FRAMES)

    if cur_mode in [ACQUIRE, HUNT, PACK]:
        for i in range(NUM_DRONES): acquire_states[i] = state_frames(acquire_state, np.zeros((1, ACQUIRE_NUM_SENSOR * ACQUIRE_STATE_FRAMES)), ACQUIRE_NUM_SENSOR, ACQUIRE_STATE_FRAMES)

    if cur_mode in [HUNT, PACK]:
        for i in range(NUM_DRONES): hunt_states[i] = state_frames(hunt_state, np.zeros((1, HUNT_NUM_SENSOR * HUNT_STATE_FRAMES)), HUNT_NUM_SENSOR, HUNT_STATE_FRAMES)

    if cur_mode == PACK:
        for i in range(NUM_DRONES): drone_states[i] = state_frames(drone_state, np.zeros((1, DRONE_NUM_SENSOR * PACK_STATE_FRAMES)),DRONE_NUM_SENSOR, PACK_STATE_FRAMES)

        pack_state = state_frames(drone_state,
                                  np.zeros((1, PACK_NUM_SENSOR * PACK_STATE_FRAMES)),
                                  PACK_NUM_SENSOR, PACK_STATE_FRAMES)
    
    cur_speeds = np.empty([NUM_DRONES, 1]); cur_speeds.fill(cur_speed)

    # time it
    start_time = timeit.default_timer()

    # run frames
    while total_frame_ctr < train_frames:
        print("1a")
        # used to slow things down for de-bugging
        #time.sleep(5)
        
        total_frame_ctr += 1 # counts total training distance traveled
        crash_frame_ctr += 1 # counts distance between crashes
        
        for drone_id in range(NUM_DRONES): # NUM_DRONES = 1, unless you're in PACK mode
            
            speed_action = START_SPEED_ACTION
            pack_action = 0
            
            # choose appropriate action(s)
            # note: only generates random inputs for currently training model.
            # all prior (sub) models provide their best (fully-trained) inputs
            if random.random() < epsilon or total_frame_ctr < observe: # epsilon degrades over flips...
                if cur_mode == TURN:
                    turn_action = set_turn_action(True, cur_speeds[drone_id],
                                                  np.array([turn_states[drone_id]]))
                else:
                    turn_action = set_turn_action(False, cur_speeds[drone_id],
                                                  np.array([turn_states[drone_id]]))
                    
                    if cur_mode == AVOID:
                        speed_action = set_avoid_action(True, turn_action,
                                                        np.array([avoid_states[drone_id]]))
                    else:
                        speed_action = set_avoid_action(False, turn_action,
                                                        np.array([avoid_states[drone_id]]))
                        
                        if cur_mode == ACQUIRE:
                            acquire_action = set_acquire_action(True, cur_speeds[drone_id],
                                                                np.array([acquire_states[drone_id,]]))
                        else:
                            acquire_action = set_acquire_action(False, cur_speeds[drone_id],
                                                                np.array([acquire_states[drone_id,]]))
                            
                            if cur_mode == HUNT:
                                hunt_action = set_hunt_action(True, cur_speeds[drone_id], turn_action,
                                                              speed_action, acquire_action,
                                                              np.array([hunt_states[drone_id,]]))
                            else:
                                hunt_action = set_hunt_action(False, cur_speeds[drone_id], turn_action,
                                                              speed_action, acquire_action,
                                                              np.array([hunt_states[drone_id,]]))
                                
                                if cur_mode == PACK and (crash_frame_ctr == 1 or crash_frame_ctr % PACK_EVAL_FRAMES == 0) and drone_id == 0:
                                    pack_action = set_pack_action(True, pack_state)
                                    # note: pack action only changed every PACK_EVAL_FRAMES.
                                    # for frames in between it's constant

            else: # ...increasing use of predictions over time
                if cur_mode == TURN:
                    turn_action = set_turn_action(False, cur_speeds[drone_id],
                                                  np.array([turn_states[drone_id]]))
                else:
                    turn_action = set_turn_action(False, cur_speeds[drone_id],
                                                  np.array([turn_states[drone_id]]))
                    
                    if cur_mode == AVOID:
                        speed_action = set_avoid_action(False, turn_action,
                                                        np.array([avoid_states[drone_id]]))
                    else:
                        speed_action = set_avoid_action(False, turn_action,
                                                        np.array([avoid_states[drone_id]]))
                        
                        if cur_mode == ACQUIRE:
                            acquire_action = set_acquire_action(False, cur_speeds[drone_id],
                                                                np.array([acquire_states[drone_id,]]))
                        else:
                            acquire_action = set_acquire_action(False, cur_speeds[drone_id],
                                                                np.array([acquire_states[drone_id,]]))
                            
                            if cur_mode == HUNT:
                                hunt_action = set_hunt_action(False, cur_speeds[drone_id], turn_action,
                                                              speed_action, acquire_action,
                                                              np.array([hunt_states[drone_id,]]))
                            else:
                                hunt_action = set_hunt_action(False, cur_speeds[drone_id], turn_action,
                                                              speed_action, acquire_action,
                                                              np.array([hunt_states[drone_id,]]))
                                                              
                                if cur_mode == PACK and (crash_frame_ctr == 1 or crash_frame_ctr % PACK_EVAL_FRAMES == 0) and drone_id == 0:
                                    # get 1 pack action for each set of drones on first drone
                                    pack_action = set_pack_action(False, pack_state)
        
            print("2a")
            # pass action, receive new state, reward
            if cur_mode == PACK: print("pack action being sent to frame step:", pack_action)
            new_turn_state, new_avoid_state, new_acquire_state, new_hunt_state, \
                new_drone_state, new_reward, new_speed = \
                game_state.frame_step(drone_id, turn_action, speed_action,
                                      pack_action, cur_speeds[drone_id], crash_frame_ctr)
            
            print("3a")
            # append (horizontally) historical states for learning speed.
            """ note: do this concatination even for models that are not learning (e.g., turn when running search or turn, search and acquire while running hunt) b/c their preds, performed above, expect the same multi-frame view that was in place when they trained."""
            if cur_mode in [TURN, AVOID, HUNT, PACK]:
                new_turn_state = state_frames(new_turn_state,
                                                     np.array([turn_states[drone_id]]),
                                                     TURN_NUM_SENSOR,TURN_STATE_FRAMES)
        
            if cur_mode in [AVOID, HUNT, PACK]:
                new_avoid_state = state_frames(new_avoid_state,
                                                      np.array([avoid_states[drone_id]]),
                                                      AVOID_NUM_SENSOR, AVOID_STATE_FRAMES)
        
            if cur_mode in [ACQUIRE, HUNT, PACK]:
                new_acquire_state = state_frames(new_acquire_state,
                                                       np.array([acquire_states[drone_id]]),
                                                       ACQUIRE_NUM_SENSOR, ACQUIRE_STATE_FRAMES)

            if cur_mode in [HUNT, PACK]:
                new_hunt_state = state_frames(new_hunt_state,
                                                     np.array([hunt_states[drone_id]]),
                                                     HUNT_NUM_SENSOR, HUNT_STATE_FRAMES)

            print("4a")
            if cur_mode == PACK and (crash_frame_ctr == 1 or crash_frame_ctr % PACK_EVAL_FRAMES == 0):
                
                if drone_id == 0: # for 1st drone, pack state = drone state
                    new_pack_state = new_drone_state
                    pack_rwd = new_reward
                
                else: # otherwise, append drone record to prior drone state
                    new_pack_state = state_frames(new_pack_state, new_drone_state,
                                                  DRONE_NUM_SENSOR, 2)
                    pack_rwd += new_reward
                
                print("pack rwd:", pack_rwd)

                new_drone_state = state_frames(new_drone_state,
                                               np.array([drone_states[drone_id]]),
                                               DRONE_NUM_SENSOR, PACK_STATE_FRAMES)

                if drone_id == (NUM_DRONES - 1): # for last drone build pack record
                    if crash_frame_ctr == 1:
                        pack_state = np.zeros((1, PACK_NUM_SENSOR * PACK_STATE_FRAMES))
                    
                    new_pack_state = state_frames(new_pack_state, pack_state, PACK_NUM_SENSOR, PACK_STATE_FRAMES) #may need to add 1 to PACK_STATE_FRAMES

                    pack_rwd = pack_rwd / NUM_DRONES

            print("*** 2. new states after move ***")
            print("crash_ctr", crash_frame_ctr, "drone_id:", drone_id)
            print(new_turn_state)
            print(new_avoid_state)
            print(new_acquire_state)
            print(new_hunt_state)
            print(new_drone_state)
            if cur_mode == PACK:
                print(new_pack_state)
                print(pack_rwd)
            
            print("2a")
            # experience replay storage
            """note: only the model being trained requires event storage as it is stack that will be sampled for training below."""
            if cur_mode == TURN:
                replay.append((turn_state, turn_action, new_reward, new_turn_state))

            elif cur_mode == AVOID:
                replay.append((avoid_state, speed_action, new_reward, new_avoid_state))

            elif cur_mode == ACQUIRE:
                replay.append((acquire_state, turn_action, new_reward, new_acquire_state))

            elif cur_mode == HUNT:
                replay.append((hunt_state, hunt_action, new_reward, new_hunt_state))

            elif cur_mode == PACK and (crash_frame_ctr == 1 or crash_frame_ctr % PACK_EVAL_FRAMES == 0) and drone_id == NUM_DRONES - 1:
                replay.append((pack_state, pack_action, new_reward, new_pack_state))

                print("*** 3. replay ***")
                print("drone id:", drone_id)
                print(replay[-1])

            print("3a")
            # If we're done observing, start training.
            if total_frame_ctr > observe and (cur_mode != PATH or (crash_frame_ctr % PACK_EVAL_FRAMES == 0 and drone_id == NUM_DRONES -1)):

                # If we've stored enough in our buffer, pop the oldest.
                if len(replay) > buffer:
                    replay.pop(0)
            
                # Randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)

                if cur_mode == TURN:
                    # Get training values.
                    X_train, y_train = process_minibatch(minibatch, turn_model,
                                                         TURN_NUM_INPUT, TURN_NUM_OUTPUT)
                    history = LossHistory()
                    turn_model.fit(X_train, y_train, batch_size=batchSize,
                                   nb_epoch=1, verbose=0, callbacks=[history])
                
                elif cur_mode == AVOID:
                    X_train, y_train = process_minibatch(minibatch, avoid_model,
                                                         AVOID_NUM_INPUT, AVOID_NUM_OUTPUT)
                    history = LossHistory()
                    avoid_model.fit(X_train, y_train, batch_size=batchSize,
                                   nb_epoch=1, verbose=0, callbacks=[history])

                elif cur_mode == ACQUIRE:
                    X_train, y_train = process_minibatch(minibatch, acquire_model,
                                                         ACQUIRE_NUM_INPUT, ACQUIRE_NUM_OUTPUT)
                    history = LossHistory()
                    acquire_model.fit(X_train, y_train, batch_size=batchSize,
                                    nb_epoch=1, verbose=0, callbacks=[history])

                elif cur_mode == HUNT:
                    X_train, y_train = process_minibatch(minibatch, hunt_model,
                                                         HUNT_NUM_INPUT, HUNT_NUM_OUTPUT)
                    history = LossHistory()
                    hunt_model.fit(X_train, y_train, batch_size=batchSize,
                                      nb_epoch=1, verbose=0, callbacks=[history])

                elif cur_mode == PACK:
                    X_train, y_train = process_minibatch(minibatch, pack_model,
                                                         PACK_NUM_INPUT, PACK_NUM_OUTPUT)
                    history = LossHistory()
                    pack_model.fit(X_train, y_train, batch_size=batchSize,
                                   nb_epoch=1, verbose=0, callbacks=[history])

                loss_log.append(history.losses)

            # Update the starting state with S'.
            turn_states[drone_id] = new_turn_state
            avoid_states[drone_id] = new_avoid_state
            acquire_states[drone_id] = new_acquire_state
            hunt_states[drone_id] = new_hunt_state
            if cur_mode == PACK and (crash_frame_ctr == 1 or crash_frame_ctr % PACK_EVAL_FRAMES == 0):
                drone_states[drone_id] = new_drone_state
                pack_state = new_pack_state
            cur_speeds[drone_id] = new_speed
            cum_rwd += new_reward

            print("*** 4. updating states ***")
            print("crash_ctr", crash_frame_ctr, "drone_id:", drone_id)
            print(turn_states)
            print(avoid_states)
            print(acquire_states)
            print(hunt_states)
            print(drone_states)
            if cur_mode == PACK: print(pack_state)
            print(cur_speeds)
            print(cum_rwd)

            # in case of crash, report and initialize
            if new_reward == -500 or new_reward == -1000:
                # Log the car's distance at this T.
                data_collect.append([total_frame_ctr, crash_frame_ctr])

                # Update max.
                if crash_frame_ctr > max_crash_frame_ctr:
                    max_crash_frame_ctr = crash_frame_ctr

                # Time it.
                tot_time = timeit.default_timer() - start_time
                fps = crash_frame_ctr / tot_time
                
                # Output some stuff so we can watch.
                print("Max: %d at %d\t eps: %f\t dist: %d\t mode: %s\t cum rwd: %d\t fps: %d" %
                      (max_crash_frame_ctr, total_frame_ctr, epsilon, crash_frame_ctr, cur_mode,
                       cum_rwd, int(fps)))

                # Reset.
                crash_frame_ctr = cum_rwd = cum_speed = 0
                start_time = timeit.default_timer()
                  
        print(9)
        # decrement epsilon for another frame
        if epsilon > 0.1 and total_frame_ctr > observe:
            epsilon -= (1/train_frames)

        if total_frame_ctr % 10000 == 0:
            if crash_frame_ctr != 0:
                print("Max: %d at %d\t eps: %f\t dist: %d\t mode: %s\t cum rwd: %d\t fps: %d" %
                      (max_crash_frame_ctr, total_frame_ctr, epsilon, crash_frame_ctr, cur_mode,
                       cum_rwd, int(fps)))
    
        # Save model every 50k frames
        if total_frame_ctr % 50000 == 0:
            save_init = False
            if cur_mode == TURN:
                turn_model.save_weights('models/turn/turn-' + filename + '-' +
                                        str(START_SPEED) + '-' + str(t) + '.h5',overwrite=True)
                print("Saving turn_model %s - %d - %d" % (filename, START_SPEED, t))
            
            elif cur_mode == AVOID:
                avoid_model.save_weights('models/avoid/avoid-' + filename + '-' +
                                         str(t) + '.h5', overwrite=True)
                print("Saving avoid_model %s - %d" % (filename, t))
            
            elif cur_mode == ACQUIRE:
                acquire_model.save_weights('models/acquire/acquire-' + filename + '-' +
                                           str(START_SPEED) + '-' + str(t) + '.h5',overwrite=True)
                print("Saving acquire_model %s - %d" % (filename, t))
            
            elif cur_mode == HUNT:
                hunt_model.save_weights('models/hunt/hunt-' + filename + '-' +
                                           str(t) + '.h5', overwrite=True)
                print("Saving hunt_model %s - %d" % (filename, t))

            elif cur_mode == PACK:
                pack_model.save_weights('models/pack/pack-' + filename + '-' +
                            str(t) + '.h5', overwrite=True)
                print("Saving pack_model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)


def set_turn_action(random_fl, cur_speed, turn_state):
    if random_fl:
        turn_action = np.random.randint(0, TURN_NUM_OUTPUT)
    else:
        if cur_mode == TURN:
            turn_qval = turn_model.predict(turn_state, batch_size=1)
            turn_action = (np.argmax(turn_qval))  # getting best prediction
        else:
            if cur_speed == SPEEDS[0]:
                turn_qval = turn_model_30.predict(turn_state, batch_size=1)
            elif cur_speed == SPEEDS[1]:
                turn_qval = turn_model_50.predict(turn_state, batch_size=1)
            elif cur_speed == SPEEDS[2]:
                turn_qval = turn_model_70.predict(turn_state, batch_size=1)
            turn_action = (np.argmax(turn_qval))
        
        return turn_action


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
    else:
        if cur_mode == ACQUIRE:
            acquire_qval = acquire_model.predict(acquire_state, batch_size=1)
            turn_action = (np.argmax(acquire_qval))
        else:
            if cur_speed == SPEEDS[0]:
                acquire_qval = acquire_model_50.predict(acquire_state, batch_size=1)
            elif cur_speed == SPEEDS[1]:
                acquire_qval = acquire_model_50.predict(acquire_state, batch_size=1)
            else:
                acquire_qval = acquire_model_70.predict(acquire_state, batch_size=1)
            turn_action = (np.argmax(acquire_qval))
        
    return turn_action


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

    return turn_action, speed_action


def set_pack_action(random_fl, pack_state):
    if random_fl:
        pack_action = np.random.randint(0, PACK_NUM_OUTPUT)
    else:
        pack_qval = pack_model.predict(pack_state, batch_size=1)
        pack_action = (np.argmax(pack_qval))

    return pack_action


def state_frames(new_state, old_state, num_sensor, num_frame):
    """
    Takes a state returned from the game and turns it into a multi-frame state.
    Create a new array with the new state and first N of old state,
    which was the previous frame's new state.
   """
    # Turn them back into arrays.
    new_state = new_state.tolist()[0]
    old_state = old_state.tolist()[0][:num_sensor * (num_frame - 1)]

    # Combine them.
    combined_state = new_state + old_state
    
    # Re-numpy them on exit.
    return np.array([combined_state])


def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def process_minibatch(minibatch, model, num_input, num_output):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        
        # Get our best move. I think?
        maxQ = np.max(newQ)
        
        y = np.zeros((1, num_output)) # was 3.
        y[:] = old_qval[:]
        
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(num_input,))
        y_train.append(y.reshape(num_output,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):

    if len(params['nn']) == 1:
    
        return str(params['nn'][0]) + '-' + str(params['batchSize']) + \
        '-' + str(params['buffer'])
    
    elif len(params['nn']) == 2:
        
        return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
        str(params['batchSize']) + '-' + str(params['buffer'])

    elif len(params['nn']) == 3:

        return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['nn'][2]) + '-' + str(params['batchSize']) + '-' + str(params['buffer'])


def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/sonar-frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/sonar-frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        if cur_mode == TURN:
            turn_model = turn_net(NUM_INPUT, params['nn'])
            train_net(turn_model, 0, params)
        elif cur_mode == AVOID:
            avoid_model = avoid_net(NUM_INPUT, params['nn'])
            train_net(0, avoid_model, params)
    
    else:
        print("Already tested.")


if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {"batchSize": batchSize, "buffer": buffer, "nn": nn_param}
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
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
                saved_model = 'models/hunt/saved/hunt-1200-240-48-100-50000-150000.h5'
                hunt_model = hunt_net(HUNT_NUM_INPUT, nn_param, HUNT_NUM_OUTPUT, saved_model)

        if cur_mode == PACK:
            print("*** in net def ***")
            print(PACK_NUM_INPUT)
            nn_param = [PACK_NUM_INPUT * 20, PACK_NUM_INPUT * 5]
            params = {"batchSize": 100, "buffer": 50000, "nn": nn_param}

            pack_model = pack_net(PACK_NUM_INPUT, nn_param, PACK_NUM_OUTPUT)

        train_net(turn_model, turn_model_30, turn_model_50, turn_model_70, avoid_model,
                  acquire_model, acquire_model_30, acquire_model_50, acquire_model_70,
                  hunt_model, pack_model, params)
