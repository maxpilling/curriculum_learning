# python -m pysc2.bin.agent \ --map Simple64 \ --agent attack_agent.SparseAgent \ --agent_race T \ --max_agent_steps 0 \ --norender
# Code segments used from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

import random
import math
import os.path

import numpy as np
import pandas as pd

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

DATA_FILE = 'sparse_agent_data'     # Used for storing the Q-Table

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



class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        # Uses epsilon greedy exploration to find the next action
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))

            action = state_action.idxmax()  # Return max value
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, next_s):
        self.check_state_exist(next_s)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':    # Not terminal meaning it does not end the game, then apply a reward
            # From next state check all actions and take the max value
            q_target = r + self.gamma * self.q_table.ix[next_s, :].max()    #NEEDS TO BE UPDATED TO USE "TD ERROR"
        else:
            q_target = r  # next state is terminal, dont apply reward

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    # Check the state exists if not we add it to the table. Makes the agent dynamically learn new states
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        # Command Center location
        self.cc_y = None
        self.cc_x = None

        # Since it is using multistep actions, we keep track of it using move_number
        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(
                DATA_FILE + '.gz', compression='gzip')

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
        super(SparseAgent, self).step(obs)

        # Checks that game has ended if so then get final reward
        if obs.last():
            reward = obs.reward     # Returned by the game: 1 if win, -1 for loss and 0 for tie (reached at 28000 steps defualt)

            self.qlearn.learn(str(self.previous_state),
                              self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

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

            # Define running stats of the player
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

            # Start by checking if first step than we learn
            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state),
                                  self.previous_action, 0, str(current_state))

            # Get action to do
            rl_action = self.qlearn.choose_action(str(current_state))

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

        return actions.FunctionCall(_NO_OP, [])

