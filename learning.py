from flat_game import carmunk
import numpy as np
import random
import csv
from nn import turn_net, avoid_net, acquire_net, hunt_net, LossHistory
import os.path
import timeit
import time

# operating modes drive neural net training
HUNT = 1
TURN = 2
AVOID = 3
ACQUIRE = 4
cur_mode = HUNT

# turn model settings
TURN_NUM_SENSOR = 10 # front five sonar distance and color readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_NUM_FRAME = 3
TURN_NUM_INPUT = TURN_NUM_FRAME * TURN_NUM_SENSOR

# avoid model settings
AVOID_NUM_SENSOR = 16 # seven sonar distance + color readings, turn action, current speed
AVOID_NUM_OUTPUT = 3 # 30, 50, 70
AVOID_NUM_FRAME = 3
AVOID_NUM_INPUT = AVOID_NUM_FRAME * AVOID_NUM_SENSOR
SPEEDS = [30,50,70]

# acquire model settings
ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_NUM_FRAME = 2
ACQUIRE_NUM_INPUT = ACQUIRE_NUM_FRAME * ACQUIRE_NUM_SENSOR

# hunt model settings
HUNT_AVOID = 0
HUNT_ACQUIRE = 1
HUNT_NUM_SENSOR = 16 # seven sonar distance + color, target distance + heading
# could go up to 19 incl turn action, speed action and acquire action
HUNT_NUM_OUTPUT = 2 # avoid (model), acquire (model)
HUNT_NUM_FRAME = 3
HUNT_NUM_INPUT = HUNT_NUM_FRAME * HUNT_NUM_SENSOR

# initial settings
use_existing_model = False
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 1
START_DISTANCE = 0

# misc
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.

def train_net(turn_model, turn_model_1, turn_model_2, turn_model_3,
              avoid_model, acquire_model, acquire_model_1, acquire_model_2,
              hunt_model, params):
    
    filename = params_to_filename(params)

    epsilon = 0.2
    train_frames = 700000  # number of flips for training
    batchSize = params['batchSize']
    buffer = params['buffer']
    
    if cur_mode == TURN:
        observe = 1000  # Number of frames to observe before training.
    elif cur_mode == AVOID:
        observe = 2000
    elif cur_mode == ACQUIRE:
        observe = 2000
    elif cur_mode == HUNT:
        observe = 1000

    # initialize variables and structures used below.
    max_car_distance = car_distance = 0
    stop_ctr = avoid_ctr = acquire_ctr = t = 0
    cum_rwd = cum_rwd_turn = cum_rwd_avoid = cum_rwd_acquire = cum_rwd_hunt = cum_speed = 0
    num_avoid_rwds = num_acquire_rwds = cum_avoid_rwds = cum_acquire_rwds = 0

    data_collect = []
    replay = []  # stores tuples of (State, Action, Reward, New State').
    save_init = True
    loss_log = []
    oversample = []

    # create game instance
    game_state = carmunk.GameState()
    
    # get initial state(s)
    turn_state, avoid_state, acquire_state, hunt_state, new_reward, cur_speed, _, _, _ = \
        game_state.frame_step(cur_mode, START_TURN_ACTION, START_SPEED_ACTION,
                              START_SPEED, START_DISTANCE)

    if cur_mode in [TURN, AVOID, HUNT]:
        turn_state = state_frames(turn_state,
                                  np.zeros((1, TURN_NUM_SENSOR * (TURN_NUM_FRAME - 1))),
                                  TURN_NUM_SENSOR, TURN_NUM_FRAME)
                                  
        if cur_mode in [AVOID, HUNT]:
            avoid_state = state_frames(avoid_state,
                                       np.zeros((1, AVOID_NUM_SENSOR * (AVOID_NUM_FRAME - 1))),
                                       AVOID_NUM_SENSOR, AVOID_NUM_FRAME)

    if cur_mode in [ACQUIRE, HUNT]:
        acquire_state = state_frames(acquire_state,
                                   np.zeros((1, ACQUIRE_NUM_SENSOR * (ACQUIRE_NUM_FRAME - 1))),
                                   ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)

    if cur_mode == HUNT:
        hunt_state = state_frames(hunt_state,
                                  np.zeros((1, HUNT_NUM_SENSOR * (HUNT_NUM_FRAME - 1))),
                                  HUNT_NUM_SENSOR, HUNT_NUM_FRAME)

    # time it
    start_time = timeit.default_timer()

    # run frames
    while t < train_frames:
        
        # used to slow things down for de-bugging
        #time.sleep(5)
        
        t += 1 # counts total training distance traveled
        car_distance += 1 # counts distance between crashes
        
        # choose an action
        speed_action = START_SPEED_ACTION
        
        if random.random() < epsilon or t < observe: # epsilon degrades over flips...
            if cur_mode in [TURN, AVOID, HUNT]:
                turn_action = np.random.randint(0, TURN_NUM_OUTPUT)
            
            if cur_mode in [AVOID, HUNT]:
                avoid_state[0][14] = turn_action # ensures avoid using current turn pred
                speed_action = np.random.randint(0, AVOID_NUM_OUTPUT)

            if cur_mode in [ACQUIRE, HUNT]:
                acquire_action = np.random.randint(0, ACQUIRE_NUM_OUTPUT)
                if cur_mode == ACQUIRE:
                    turn_action = acquire_action
            
            if cur_mode == HUNT:
                hunt_action = np.random.randint(0, HUNT_NUM_OUTPUT)
                
                if hunt_action == HUNT_AVOID: # accept avoid model action
                    turn_action = turn_action
                    if cur_speed > 0:
                        speed_action = speed_action # continue current speed
                    else:
                        speed_action = SPEEDS[1] # reset speed to 50, as you were stopped
                    avoid_ctr += 1
                
                elif hunt_action == HUNT_ACQUIRE: # accept acquire model action
                    turn_action = acquire_action
                    if cur_speed > 0:
                        speed_action = SPEEDS[1] # just setting acquire speed to 50 for now
                    else:
                        speed_action = SPEEDS[1] # reset acquire speed to 50, as you were stopped
                    acquire_ctr += 1
    
        else: # ...increasing use of predictions over time
            #print("*************** new rcd ***************")
            if cur_mode == TURN:
                turn_qval = turn_model.predict(turn_state, batch_size=1)
                turn_action = (np.argmax(turn_qval))  # getting best prediction
            
            if cur_mode in [AVOID, HUNT]:
                if cur_speed == SPEEDS[0]:
                    turn_qval = turn_model_1.predict(turn_state, batch_size=1)
                elif cur_speed == SPEEDS[1]:
                    turn_qval = turn_model_2.predict(turn_state, batch_size=1)
                elif cur_speed == SPEEDS[2]:
                    turn_qval = turn_model_3.predict(turn_state, batch_size=1)
                #elif cur_speed in [SPEEDS[3]]:
                    #turn_qval = turn_model_3.predict(turn_state, batch_size=1)
                turn_action = (np.argmax(turn_qval))
                #print("turn action:", turn_action)
                avoid_state[0][14] = turn_action
                avoid_qval = avoid_model.predict(avoid_state, batch_size=1)
                speed_action = (np.argmax(avoid_qval))
                #print("avoid action:", turn_action)
    
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
                #print("acquire action:", acquire_action)
                # may want to pred hunt action based on speed turn/speed actions and acquire turn action
                # right now it's deciding just based on distances and headings (obstacles, target)
                hunt_qval = hunt_model.predict(hunt_state, batch_size=1)
                hunt_action = (np.argmax(hunt_qval))
                #print("hunt action:", hunt_action)
                
                if hunt_action == HUNT_AVOID: # accept avoid model action
                    turn_action = turn_action
                    if cur_speed > 0:
                        speed_action = speed_action # continue current speed
                    else:
                        speed_action = 2
                    avoid_ctr += 1
                
                elif hunt_action == HUNT_ACQUIRE: # accept acquire model action
                    turn_action = acquire_action
                    if cur_speed > 0:
                        speed_action = 1 # just setting acquire speed to 50 for now
                    else:
                        speed_action = 1 # reset acquire speed to 50, as you were stopped
                    acquire_ctr += 1
        
        # pass action, receive new state, reward
        new_turn_state, new_avoid_state, new_acquire_state, new_hunt_state, new_reward, new_speed, new_rwd_turn, new_rwd_acquire, new_rwd_avoid = game_state.frame_step(cur_mode, turn_action, speed_action, cur_speed, car_distance)
        
        #print("new hunt state:")
        #print(new_hunt_state)
        
        # append (horizontally) historical states for learning speed.
        """ note: do this concatnation even for models that are not learning (e.g., turn when running search or turn, search and acquire while running hunt) b/c their preds, performed above, expect the same multi-frame view that was in place when they trained."""
        if cur_mode in [TURN, AVOID, HUNT]:
            new_turn_state = state_frames(new_turn_state, turn_state,
                                          TURN_NUM_SENSOR,TURN_NUM_FRAME)
        
        if cur_mode in [AVOID, HUNT]:
            new_avoid_state = state_frames(new_avoid_state, avoid_state,
                                           AVOID_NUM_SENSOR, AVOID_NUM_FRAME)
        
        if cur_mode in [ACQUIRE, HUNT]:
            new_acquire_state = state_frames(new_acquire_state, acquire_state,
                                             ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)

        if cur_mode == HUNT:
            new_hunt_state = state_frames(new_hunt_state, hunt_state,
                                          HUNT_NUM_SENSOR, HUNT_NUM_FRAME)

        # experience replay storage
        """note: only the model being trained requires event storage as it is stack that will be sampled for training below."""
        if cur_mode == TURN:
            replay.append((turn_state, turn_action, new_reward, new_turn_state))

        elif cur_mode == AVOID:
            replay.append((avoid_state, speed_action, new_reward, new_avoid_state))

        elif cur_mode == ACQUIRE:
            replay.append((acquire_state, turn_action, new_reward, new_acquire_state))

        elif cur_mode == HUNT:
            if hunt_action == HUNT_AVOID:
                new_reward = new_rwd_avoid
                num_avoid_rwds +=1
                cum_avoid_rwds += new_rwd_avoid
            elif hunt_action == HUNT_ACQUIRE:
                new_reward = new_rwd_acquire
                num_acquire_rwds +=1
                cum_acquire_rwds += new_rwd_acquire
            
            replay.append((hunt_state, hunt_action, new_reward, new_hunt_state))

        # If we're done observing, start training.
        if t > observe:
  
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

            loss_log.append(history.losses)

        # Update the starting state with S'.
        turn_state = new_turn_state
        avoid_state = new_avoid_state
        acquire_state = new_acquire_state
        hunt_state = new_hunt_state
        cur_speed = new_speed
        
        # accumulations
        cum_rwd += new_reward
        cum_rwd_turn += new_rwd_turn
        cum_rwd_acquire += new_rwd_acquire
        cum_rwd_avoid += new_rwd_avoid
        cum_speed += cur_speed

        # Decrement epsilon over time.
        if epsilon > 0.0 and t > observe:
            epsilon -= (1/(train_frames/2))
        
        # We died, so update stuff.
        if new_reward == -500 or new_reward == -1000:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time
            
            # Output some stuff so we can watch.
            print("Max: %d at %d\t eps: %f\t dist: %d\t rwd: %d\t turn: %d\t acq: %d\t spd: %d\t avg spd: %f\t fps: %d" %
                  (max_car_distance, t, epsilon, car_distance, cum_rwd, cum_rwd_turn,
                   cum_rwd_acquire, cum_rwd_avoid, cum_speed / car_distance, int(fps)))

            # Reset.
            car_distance = 0
            cum_rwd = 0
            cum_rwd_turn = 0
            cum_rwd_acquire = 0
            cum_rwd_avoid = 0
            cum_rwd_hunt = 0
            cum_speed = 0
            start_time = timeit.default_timer()

        # oversampling: tripling odds that success record is selected
        #if cur_mode == ACQUIRE and new_reward == 1000:
        #    replay.append((acquire_state, turn_action, new_reward, new_acquire_state))
        #    replay.append((acquire_state, turn_action, new_reward, new_acquire_state))

        if t % 10000 == 0:
            #if car_distance != 0:
            #    print("t: %d\t eps: %f\t dist: %d\t rwd: %d\t turn: %d\t acq: %d\t spd: %d\t avg spd: %f\t hunt: %d" %
            #      (t, epsilon, car_distance, cum_rwd, cum_rwd_turn, cum_rwd_acquire,
            #       cum_rwd_avoid, cum_speed / car_distance, cum_rwd_hunt))

            if cur_mode == HUNT:
                print("avoid ctr:", avoid_ctr, "acquire ctr:", acquire_ctr)
                stop_ctr = avoid_ctr = acquire_ctr = 0
                print("num spd rwd:", num_avoid_rwds, "cum spd rwds:", cum_avoid_rwds)
                print("num acq rwd:", num_acquire_rwds, "cum acq rwds:", cum_acquire_rwds)

        # Save early turn_model, then every 20,000 frames
        if t % 50000 == 0:
            save_init = False
            if cur_mode == TURN:
                turn_model.save_weights('models/turn/turn-' + filename + '-' +
                                        str(START_SPEED) + '-' + str(t) + '.h5',
                                        overwrite=True)
                print("Saving turn_model %s - %d - %d" % (filename, START_SPEED, t))
            
            elif cur_mode == AVOID:
                avoid_model.save_weights('models/avoid/avoid-' + filename + '-' +
                                         str(t) + '.h5', overwrite=True)
                print("Saving avoid_model %s - %d" % (filename, t))
            
            elif cur_mode == ACQUIRE:
                acquire_model.save_weights('models/acquire/acquire-' + filename + '-' +
                                           str(START_SPEED) + '-' + str(t) + '.h5',
                                           overwrite=True)
                print("Saving acquire_model %s - %d" % (filename, t))
            
            elif cur_mode == HUNT:
                hunt_model.save_weights('models/hunt/hunt-' + filename + '-' +
                                           str(t) + '.h5', overwrite=True)
                print("Saving hunt_model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)


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
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
        turn_model = turn_model_1 = turn_model_2 = turn_model_3 = \
        avoid_model = acquire_model = acquire_model_1 = acquire_model_2 = \
        hunt_model = 0
        
        if cur_mode == TURN:
            nn_param = [TURN_NUM_INPUT * 25, TURN_NUM_INPUT * 10]
            # 3-layer [25x, 5x, x] performed horribly w/ and w/out color
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
            turn_model = turn_net(TURN_NUM_INPUT, nn_param, TURN_NUM_OUTPUT)

        elif cur_mode == AVOID:
            saved_model = 'models/turn/saved/turn-750-300-100-50000-30-600000.h5'
            turn_model_1 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                     TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-50-600000.h5'
            turn_model_2 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                     TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-70-600000.h5'
            turn_model_3 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                     TURN_NUM_OUTPUT, saved_model)

            nn_param = [AVOID_NUM_INPUT * 25, AVOID_NUM_INPUT * 5, AVOID_NUM_INPUT]
            #nn_param = [AVOID_NUM_INPUT * 25, AVOID_NUM_INPUT * 5, AVOID_NUM_INPUT]
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
    
            if use_existing_model == True:
                saved_model = 'models/avoid/saved/avoid-1200-240-100-50000-650000-old-3L-2022.h5'
                avoid_model = avoid_net(AVOID_NUM_INPUT, nn_param, AVOID_NUM_OUTPUT, saved_model)
            
            else:
                avoid_model = avoid_net(AVOID_NUM_INPUT, nn_param, AVOID_NUM_OUTPUT)

        elif cur_mode == ACQUIRE:
            nn_param = [ACQUIRE_NUM_INPUT * 15, ACQUIRE_NUM_INPUT * 5]
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
            acquire_model = acquire_net(ACQUIRE_NUM_INPUT, nn_param, ACQUIRE_NUM_OUTPUT)

        elif cur_mode == HUNT:
            saved_model = 'models/turn/saved/turn-750-300-100-50000-30-600000.h5'
            turn_model_1 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-50-600000.h5'
            turn_model_2 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                    TURN_NUM_OUTPUT, saved_model)
            saved_model = 'models/turn/saved/turn-750-300-100-50000-70-600000.h5'
            turn_model_3 = turn_net(TURN_NUM_INPUT, [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10],
                                                            TURN_NUM_OUTPUT, saved_model)

            saved_model = 'models/avoid/saved/avoid-1200-240-48-100-50000-700000-old-3L-2022.h5'
            avoid_model = avoid_net(AVOID_NUM_INPUT,
                                    [AVOID_NUM_INPUT * 25, AVOID_NUM_INPUT * 5, AVOID_NUM_INPUT],
                                    AVOID_NUM_OUTPUT, saved_model)
    
            saved_model = 'models/acquire/acquire-60-20-100-50000-50-350000.h5'
            acquire_model_1 = acquire_net(ACQUIRE_NUM_INPUT,
                                          [ACQUIRE_NUM_INPUT*15, ACQUIRE_NUM_INPUT*5],
                                          ACQUIRE_NUM_OUTPUT, saved_model)
            saved_model = 'models/acquire/acquire-60-20-100-50000-70-350000.h5'
            acquire_model_2 = acquire_net(ACQUIRE_NUM_INPUT,
                                          [ACQUIRE_NUM_INPUT*15, ACQUIRE_NUM_INPUT*5],
                                          ACQUIRE_NUM_OUTPUT, saved_model)
                
            nn_param = [HUNT_NUM_INPUT * 25, HUNT_NUM_INPUT * 5, HUNT_NUM_INPUT]
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
            if use_existing_model == True:
                saved_model = 'models/hunt/hunt-1200-240-48-100-50000-125to100-300000.h5'
                hunt_model = hunt_net(HUNT_NUM_INPUT, nn_param, HUNT_NUM_OUTPUT, saved_model)

            else:
                hunt_model = hunt_net(HUNT_NUM_INPUT, nn_param, HUNT_NUM_OUTPUT)
                                            

        train_net(turn_model, turn_model_1, turn_model_2, turn_model_3,
                  avoid_model, acquire_model, acquire_model_1, acquire_model_2,
                  hunt_model, params)
