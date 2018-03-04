# python -m pysc2.bin.agent \ --map Simple64 \ --agent DeepQ.Agent \ --agent_race T \ --max_agent_steps 0 \ --norender
# Code segments used from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

import random
import math
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]   # Allows the selecting of all units of a certain type

#DATA_FILE = 'agent_data'     # NEED TO UPDATE FOR SAVING SESSION GRAPH

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
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            # Create every possible attack position. Will look like: "attack_5_10"
            smart_actions.append(ACTION_ATTACK + '_' +
                                 str(mm_x - 16) + '_' + str(mm_y - 16)) # Subtract 16 bec we want to select the top left corner of grid



class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        self.actions = actions  # list of int
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.memory = []    # Used to store the memory of each game step taken

        # ------ Setup NN ---------
        self.n_input = 8           # Number of nodes on first layer (the input)
        self.n_hidden1 = 20        # Number of nodes on hidden layer 1
        self.n_hidden2 = 10        # Number of nodes on hidden layer 2
        self.n_target = 8          # Number of nodes on final layer (the output)


        """ 
            A better method would to implement an output layer with an putput per action. 
            This reduces the number of feedforward passes needed and improves the overall performance.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_input], name='Input')         # Create 16 nodes input array for states and actions
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_target], name='output')                           # Create output node which is a single value

            # Create Weights
            #self.W1 = tf.Variable(tf.random_uniform(shape=[self.n_input, self.n_hidden1], minval=0, seed=None, dtype=tf.float32), dtype=tf.float32)    # 8 inputs and 1 output
            self.W1 = tf.Variable(tf.random_normal(shape=[self.n_input, self.n_hidden1], dtype=tf.float32), dtype=tf.float32)       # Weights for first hidden layer
            self.b1 = tf.Variable(tf.zeros([self.n_hidden1]), name='b1') 

            self.W2 = tf.Variable(tf.random_normal(shape=[self.n_hidden1, self.n_hidden2], dtype=tf.float32), dtype=tf.float32)
            self.b2 = tf.Variable(tf.zeros([self.n_hidden2]), name='b2')
            
            self.W_out = tf.Variable(tf.random_normal(shape=[self.n_hidden2, self.n_target], dtype=tf.float32), dtype=tf.float32)
            self.b_out = tf.Variable(tf.zeros([self.n_target]), name='b_out')

            # Connect the nodes
            self.hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.W1), self.b1))
            self.hidden_2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_1, self.W2), self.b2))

            #self.hidden_1 = tf.nn.relu(tf.matmul(self.X, self.W1))
            #self.hidden_2 = tf.nn.relu(tf.matmul(self.hidden_1, self.W2))

            self.Qout = tf.add( tf.matmul(self.hidden_2, self.W_out), self.b_out)   # Multiply weights by nodes
            #self.Qout = tf.matmul(self.hidden_2, self.W_out)

            self.loss = tf.reduce_sum(tf.square(self.y - self.Qout))    # Loss function

            self.trainer = tf.train.GradientDescentOptimizer(self.lr)   # Update the network
            self.updateModel = self.trainer.minimize(self.loss)         # Minimize loss
            self.predict = tf.argmax(self.Qout, 1)                      # Predicted value
            
            self.init = tf.global_variables_initializer()               # Initilizes variables
            self.saver = tf.train.Saver()                               # Saves the model and restores

        # Need to initialize variabless
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run(self.init)

    def choose_action(self, observation):
        
        # Uses epsilon greedy exploration to find the next action
        if np.random.uniform() < self.epsilon:
            with self.sess.as_default():
                #choose best action
                states = observation                                # Take states           
                states_T = np.reshape(states, (-1, self.n_input) )  # Reshape to make 9 rows instead

                output, allQ = self.sess.run( [ self.predict, self.W1 ], feed_dict={self.X: states_T }) # Run network and get value for action in state

            return output[0]       # Return index with max value action

        else:
            # choose random action
            action = np.random.randint(low=0, high=7)  # Take a random action
            return action

    """ Adds the state, action, reward recieved, and next state is terminal or normal state. """
    def remember(self, s, a, r, next_s):
        
        # Save to memory 
        self.memory.append([s, a, r, next_s])

    """ Update the wieghts based on the memory. """
    def learn(self):
        # May not be able to use minibatch since we need to update the q target in every run
        
        action_values = []

        with self.sess.as_default():

                for mem in self.memory:
                    # Get prediction
                    states_P = np.reshape(mem[0], (-1, self.n_input) )  # Reshape to make 8 rows instead
                    output_P, allQ_P = self.sess.run([self.predict, self.Qout], feed_dict={self.X: states_P })
                    q_target = allQ_P

                    # If not a terminal state then look at next state action value for Q target
                    if mem[3].any() != [-1] :
                        states_T = np.reshape(mem[3], (-1, self.n_input) )  # Reshape to make 8 rows instead

                        output, allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.X: states_T })  # Get the next states max action value by running the network
                        # Q Learning update
                        #allQ[0, output[0]] =  mem[2] + ( self.gamma * allQ[0, output[0]] )
                        q_target[0, output_P[0]] = mem[2] + ( self.gamma * allQ[0, output[0]] )

                    # If ternminal state
                    else:
                        # Assign reward since there is no next state in terminal
                        q_target[0, output_P[0]] = mem[2]

                    updateM, updateL = self.sess.run( [self.updateModel, self.loss] , feed_dict={self.X: states_P, self.y: q_target })
        self.memory = []

class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()

        self.qlearn = QLearning(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

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

        self.turn_counter += 1

        # Checks that game has ended if so then get final reward
        if obs.last():
            reward = obs.reward     # Returned by the game: 1 if win, -1 for loss and 0 for tie (reached at 28000 steps defualt)

            self.qlearn.remember(self.previous_state, self.previous_action, reward, next_s=[-1])
            self.qlearn.learn()

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        # Checks if its the first step of the game then create required items
        if obs.first():
            # Player y and x are a list of the agents units in pixels covered by the unit
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

            # Take from list the mean value and check position. If mean is greater than 31 chances are the units are on bottom right
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

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
            current_state = np.zeros(8)
            current_state[0] = cc_count                                 # Number of command centers
            current_state[1] = supply_depot_count                       # Number of supply depots
            current_state[2] = barracks_count                           # Number of barracks
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]  # Army supply 
            
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

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            # Save to memory for learning later
            if self.previous_action is not None:
                r = 0                                           # R is rewards for building and army

                if current_state[1] > self.previous_state[1]:   # Check for new supply depots built
                    r += 5*current_state[1]
                if current_state[2] > self.previous_state[2]:   # Check for new barracks built
                    r += 5*current_state[2]
                if current_state[3] > self.previous_state[3]:   # Check for new army units
                    r += 6*current_state[3]
                if current_state[4] < self.previous_state[4] or current_state[5] < self.previous_state[5] or current_state[6] < self.previous_state[6] or current_state[7] < self.previous_state[7]:   # Check for any enemy kills
                    r += 15

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
                    target = [unit_x[i], unit_y[i]]         # Pick a random one

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])   # Select the unit
            
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
                if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:  # We want to build 2 depots
                    if self.cc_y.any():
                        if supply_depot_count == 0:
                            # Build them in fixed locations
                            target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)

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
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])

        # Increment mover counter. We know ensure SCV goes back to work after building 
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
        
        # Check counter to learn and reset every 20 steps
        if self.turn_counter > 50:        
            self.qlearn.learn()
            self.turn_counter = 0

        return actions.FunctionCall(_NO_OP, [])


