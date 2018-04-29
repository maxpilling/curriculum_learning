# To run the agent and game: python -m pysc2.bin.agent \ --map Simple64 \ --agent DeepQ.Agent \ --agent_race T \ --max_agent_steps 0 \ --norender
# Code segments used from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

import random
import math
import os.path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

LEARN = True    # Change to make agent learn or play without learning

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

# State relative info
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5
_SUPPLY_LIMIT = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]   # Allows the selecting of all units of a certain type


ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

# Since we are using the minimap we would have a 64x64 grid
# Break the entire map into 4 quads
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))   # Subtract to aim for the center of the quads

""" The Deep Q Learning Class uses a neural network to evaluate state and actions. """
class QLearning:
    def __init__(self, actions, learning_rate=0.001, reward_decay=0.9, e_greedy=0.3):
        self.actions = actions  # list of int
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay = 0.999

        self.memory = []    # Used to store the memory of each game step taken

        # ------ Setup NN ---------
        self.n_input = 13          # Number of nodes on first layer (the input)
        self.n_hidden1 = 200        # Number of nodes on hidden layer 1
        self.n_hidden2 = 300        # Number of nodes on hidden layer 2
        self.n_hidden3 = 200        # Number of nodes on hidden layer 3
        self.n_hidden4 = 100        # Number of nodes on hidden layer 4
        self.n_target = 8         # Number of nodes on final layer (the output)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='Input')         # Create 16 nodes input array for states and actions
            self.Y = tf.placeholder(tf.float32, shape=[None, self.n_target], name='output')                           # Create output node which is a single value

            # Create Weights
            self.W1 = tf.Variable(tf.random_normal(shape=[self.n_input, self.n_hidden1], dtype=tf.float32, stddev=0.5), dtype=tf.float32)       # Weights for first hidden layer
            self.b1 = tf.Variable(tf.zeros([self.n_hidden1]), name='b1') 

            self.W2 = tf.Variable(tf.random_normal(shape=[self.n_hidden1, self.n_hidden2], dtype=tf.float32, stddev=0.5), dtype=tf.float32)
            self.b2 = tf.Variable(tf.zeros([self.n_hidden2]), name='b2')
            
            self.W3 = tf.Variable(tf.random_normal(shape=[self.n_hidden2, self.n_hidden3], dtype=tf.float32, stddev=0.5), dtype=tf.float32)
            self.b3 = tf.Variable(tf.zeros([self.n_hidden3]), name='b3')

            self.W4 = tf.Variable(tf.random_normal(shape=[self.n_hidden3, self.n_hidden4], dtype=tf.float32, stddev=0.5), dtype=tf.float32)
            self.b4 = tf.Variable(tf.zeros([self.n_hidden4]), name='b4')

            self.W_out = tf.Variable(tf.random_normal(shape=[self.n_hidden4, self.n_target], dtype=tf.float32, stddev=0.5), dtype=tf.float32)
            self.b_out = tf.Variable(tf.zeros([self.n_target]), name='b_out')

            # Connect the nodes
            self.hidden_1 = tf.nn.relu( tf.add(tf.matmul(self.X, self.W1), self.b1 ))
            self.hidden_2 = tf.nn.relu( tf.add(tf.matmul(self.hidden_1, self.W2), self.b2 ))
            self.hidden_3 = tf.nn.relu( tf.add(tf.matmul(self.hidden_2, self.W3), self.b3 ))
            self.hidden_4 = tf.nn.relu( tf.add(tf.matmul(self.hidden_3, self.W4), self.b4 ))

            self.Qout = tf.add( tf.matmul(self.hidden_4, self.W_out), self.b_out)   # Multiply weights by nodes

            self.loss = tf.reduce_sum(tf.abs(self.Y - self.Qout))    # Loss function

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

            self.predict = tf.argmax(self.Qout, 1)                      # Predicted value
            
            self.init = tf.global_variables_initializer()               # Initilizes variables
            self.saver = tf.train.Saver()                               # Saves the model and restores

        # Need to initialize variabless or import variables from existing file
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            if os.path.exists("./model"):
                self.saver.restore(self.sess, "./model/agent_model.ckpt")
            else:
                self.sess.run(self.init)

    def choose_action(self, observation):
        
        # Uses epsilon greedy exploration to find the next action
        if np.random.uniform() > self.epsilon:
            with self.sess.as_default():
                #choose best action
                states = observation                                # Take states           
                states_T = np.reshape(states, (-1, self.n_input) )  # Reshape to make rows instead

                output, allQ = self.sess.run( [ self.predict, self.Qout ], feed_dict={self.X: states_T }) # Run network and get value for action in state
                return output[0]       # Return index with max value action

        else:
            if LEARN == True:
                # choose random action
                action = np.random.randint(low=0, high=self.n_target-1)  # Take a random action

                # The epsilon value for a random action
                if self.epsilon > .011:
                    self.epsilon *= self.epsilon_decay         # Reduces the value of epsilon everytime we take a random step

                return action
            else:
                return 0

    """ Adds the state, action, reward recieved, and if next state is terminal or normal state. """
    def remember(self, s, a, r, next_s):
        
        # Save to memory 
        self.memory.append([s, a, r, next_s])

    """ Updates the wieghts based on the memory array. """
    def learn(self):

        with self.sess.as_default():

            total_loss = 0  # Accumulates the total loss of a run
            sample_memory = np.random.permutation(self.memory)

            for mem in sample_memory:
                # Get prediction
                states_P = np.reshape(mem[0], (-1, self.n_input) )  # Reshape to make 8 rows instead
                output_P, allQ_P = self.sess.run([self.predict, self.Qout], feed_dict={self.X: states_P })
                q_target = allQ_P

                # If not a terminal state then look at next state action value for Q target
                if not any(x in mem[3] for x in [-1]):
                    states_T = np.reshape(mem[3], (-1, self.n_input) )  # Reshape to make 8 rows instead

                    output, allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.X: states_T })  # Get the next states max action value by running the network
                    # Q Learning update
                    q_target[0, mem[1]] = mem[2] + ( self.gamma * allQ[0, output[0]] )

                # If ternminal state
                else:
                    # Assign reward since there is no next state in terminal
                    q_target[0, mem[1]] = mem[2]

                updateM, updateL = self.sess.run( [self.optimizer, self.loss] , feed_dict={self.X: states_P, self.Y: q_target })
                total_loss += updateL   # Accumulate
        # Get average loss
        total_loss = total_loss / len(self.memory)
        self.memory = []

        # Save the total loss to a file for analysis
        with open("loss.txt", "a") as myfile:
            myfile.write( str(total_loss) + "," )

class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()

        self.qlearn = QLearning(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_score = 0

        # Command Center location
        self.cc_y = None
        self.cc_x = None

        # Since it is using multistep actions, we keep track of it using move_number
        self.move_number = 0

        # Step counter for learning every certain count
        self.turn_counter = 0

    # Used for the case when base is at bottom right
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    # Used for the case when base is at bottom right
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    # Get the action and location
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def step(self, obs):
        super(Agent, self).step(obs)
        
        # Checks that game has ended if so then get final reward
        if obs.last():
            # When using simple 64 map use this reward
            reward = obs.reward     # Returned by the game: 1 if win, -1 for loss and 0 for tie (reached at 28000 steps defualt)

            # When using minigames use below reward for score          
            # reward = obs.observation['score_cumulative'][0]

            # Use a negative reward for some tasks
            # reward = 0
            # if obs.observation['score_cumulative'][0] > self.previous_score:
            #     reward = 1
            # else:
            #     reward = -0.5

            if LEARN == True:
                self.qlearn.remember(self.previous_state, self.previous_action, reward, [-1])
                self.qlearn.learn()
                # Save the model to a folder
                with self.qlearn.sess.as_default(): 
                    self.qlearn.saver.save(self.qlearn.sess, "./model/agent_model.ckpt")

            # Reset values
            self.previous_action = None
            self.previous_state = None

            self.move_number = 0
            self.previous_score = 0
            
            # Save the score to a file
            with open("score.txt", "a") as scoreFile:
                # scoreFile.write( str(obs.observation['score_cumulative'][0]) + "," )  # Use for logging in-game score
                scoreFile.write( str(obs.reward) + "," )    # Use for logging final win, draw or loss

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        # Checks if its the first step of the game then create required items
        if obs.first():
            # Player y and x are a list of the agents units in pixels covered by the unit
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

            # Take from list the mean value and check position. If mean is greater than 31 chances are the units are on bottom right
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            # self.base_top_left = 1 # Make the agent always play minigames with base top left

            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        # We need to count how many of each we have. In this case we get the length of the pixels covered by the unit and divide by its pixel size
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))                  # Divide by the number of pixels a depot usually covers "69" in this case

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))                  # Divide by the number of pixels a command center usually takes "137" in this case

        if self.move_number == 0:
            self.move_number += 1

            # Define running stats of the player. This is the state of of the game for the agent
            current_state = np.zeros(13)
            current_state[0] = cc_count*0.1                                 # Number of command centers
            current_state[1] = supply_depot_count*0.1                       # Number of supply depots
            current_state[2] = barracks_count*0.1                           # Number of barracks
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]*0.1  # Army supply 
            current_state[4] = obs.observation['player'][_SUPPLY_LIMIT]*0.1 # Supply limit
        
            # Hot squares defines location where enemies are. We divide map into 4 quadrants and mark each with 1 if enemy found
            hot_squares = np.zeros(4)
            # Get a list of hostile units locations
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

            for i in range(0, len(enemy_y)): 
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, len(hot_squares)):
                current_state[i + 5] = hot_squares[i]

            # Get position of minereals
            shards_y, shards_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_NEUTRAL).nonzero()
            hot_squares = np.zeros(4)

            for i in range(0, len(shards_y)):
                y = int(math.ceil((shards_y[i] + 1) / 32))
                x = int(math.ceil((shards_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1    # Center the attack to the middle coordinate so agent attacks surrounding

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, len(hot_squares)):
                current_state[i + 9] = hot_squares[i]       # +8 bec we already have 8 states before

            # Save state, action, reward and next state to memory
            if self.previous_action is not None:
                r = 0      # R is rewards from enviroment
            
                # Using cumulative score we can tell if agent killed or destroyed a building

                #killed_unit_score = obs.observation['score_cumulative'][5]
                #killed_building_score = obs.observation['score_cumulative'][6]

                #if killed_unit_score > self.previous_killed_unit_score:
                #    r += 0.5
            
                #if killed_building_score > self.previous_killed_building_score:
                #    r += 0.5

                # Store new cumulative score
                #self.previous_killed_unit_score = killed_unit_score
                #self.previous_killed_building_score = killed_building_score

                # For mini game rewards
                r = obs.observation['score_cumulative'][0] - self.previous_score

                # Below are different rewards that can be used
                # if obs.observation['score_cumulative'][0] > self.previous_score:
                #     #r = obs.observation['score_cumulative'][0]
                #     r = 1
                # else:
                #     r = -0.5

                self.previous_score = obs.observation['score_cumulative'][0]

                if LEARN == True:
                    self.qlearn.remember(self.previous_state, self.previous_action, r, current_state)
        
            # Get action to do
            rl_action = self.qlearn.choose_action(current_state)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            """ All actions will be 3 game steps: 
                - The selection of a unit
                - The action to take
                - A follow up action
            """
            # Build Barracks
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()   # Get list of scv units

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]         # Pick a random SCV

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])   # Select the SCV unit
        
            # Build Marine
            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target]) # Use select all to choose all barracks. The game auto chooses empty barracks to build units in to balance workload
        
            # Select the army
            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        # Increment into next move stage
        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action) # Get next move

            # Build Supply Depot
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:  # We want to build 2 depots
                    if self.cc_y.any():
                        target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), np.random.randint(-30, 30))

                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            # Build Barracks
            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:  # Build only 2
                    if self.cc_y.any():
                        # Build them in fixed locations
                        if barracks_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9) 
                        elif barracks_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        
            # Build Marine
            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            # Attack with marines
            elif smart_action == ACTION_ATTACK:
                do_it = True

                # Make sure no SCV selected
                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                # Make sure no SCV selected with army
                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                # Find random quad to attack. REMEMBER that attacking a point attacks 4 surrounding quads. So we choose the center of a quad and this will attack surrounding area
                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation( int(x), int(y) )])

        # Increment mover counter. We ensure SCV goes back to work after building 
        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            # We check that we had sent an SCV to do the building
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                # Check it can harvest
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        # Get random harvest location
                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target]) # Send SCV to harvest. NOTICE it is queued so SCV will finish building first
    
        return actions.FunctionCall(_NO_OP, [])


