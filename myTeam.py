# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import json
import ujson

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLearningAgent', second='DefensiveReflexAgent', num_training=5):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class QLearningCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.qvalues = util.Counter()
        self.qvalues_path = 'qvalues.json'
        self.epsilon = 0.4
        self.discount = 0.8
        self.alpha = 0.2
        

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.gridHeight = game_state.data.layout.height
        self.gridWidth = game_state.data.layout.width

        with open(self.qvalues_path, 'r') as f:
            self.qvalues = ujson.load(f)
        """
        print("Qvalues dictionary file content:")
        for key, value in self.qvalues.items():
            print (key, value)
        """

    def get_qvalue(self, position, action):
        # Returns the current qvalue stored in counter/dictionary for this position and action
        # If not found, returns 0.0 as the default value

        qvalue = self.qvalues.get((position, action), 0.0)
        return qvalue
                
    def computeValueFromQValues(self, game_state):
        """
          Returns max qvalue over the legal actions
          Returns 0.0 if there are no legal actions,
          which is the case at the terminal state
        """
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return 0.0
        # Get the agent's position
        myState = game_state.get_agent_state(self.index)
        myPosition = myState.get_position()
        # get the list of the qvalues for all the legal actions
        values = [self.get_qvalue(myPosition, action) for action in legalActions]
        # get the maximum qvalue within the list
        maxValue = max(values)

        return maxValue

    def computeActionFromQValues(self, game_state):
        """
          returns one of the actions with the highest qvalue
          If there are no legal actions, which is the case at the terminal state,
          it returns None.
        """
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None
        # Get the agent's position
        myState = game_state.get_agent_state(self.index)
        myPosition = myState.get_position()
        # get the list of the qvalues for all the legal actions
        values = [self.get_qvalue(myPosition, action) for action in legalActions]
        # get the maximum qvalue within the list
        maxValue = max(values)

        # Get a list of all the legal actions with this same qvalue
        bestActionsList = [action for action, value in zip(legalActions, values) if value == maxValue]

        # Return a randomly cvhosen action among the best actions list
        return random.choice(bestActionsList)

    def choose_action(self, game_state):
        """
          Choose the action to take in the current state.  With probability self.epsilon,
          we ignore qvalues and simply take a random action.
          Otherwise we take one of the best policy action.
          If there are no legal actions, which is the case at the terminal state,  we return
          None as the action.
        """
        # Pick Action
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        # Get the best action according to the state and the current qvalues
        action = self.computeActionFromQValues(game_state)

        # Update the qvalues dictionary according to the current state,
        # the last action that will be taken, the next position and the reward
        # obtained by going to this new position
        if action != None:
            nextState = self.get_successor(game_state, action)
            reward = self.get_reward(game_state, action)
            self.update(game_state, action, nextState, reward)

        return action

    def update(self, game_state, action, nextState, reward):
        """
          Computes the new qvalue for the agent's current state and its next action
          Then update the qvalues dictionary with this new value
        """

        # Get the agent's current position
        myState = game_state.get_agent_state(self.index)
        myPosition = myState.get_position()

        stateQValue = self.get_qvalue(myPosition, action)
        #print("Position: ", myPosition, " action: ", action, " current QValue: ", stateQValue)
        currentPart = (1-self.alpha) * stateQValue
        nextValue = self.computeValueFromQValues(nextState)
        #print("New Value: ", nextValue)
        expectedNext = self.alpha * (reward + self.discount * nextValue)
        nextQValue = currentPart + expectedNext
        #print("Reward: ", reward, " new Qvalue: ", nextQValue)
        self.qvalues[(myPosition, action)] = nextQValue
        """ 
        print("Qvalues dictionary file content:")
        for key, value in self.qvalues.items():
            print(key, value)
        """

        with open(self.qvalues_path, 'w') as f:
            f.write(ujson.dumps(self.qvalues))

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def dist_to_food_to_eat(self, state):
        # Returns the distance to the nearest food of the opponent team
        # ( i.e. the food the agent has to eat)
        myState = state.get_agent_state(self.index)
        myPosition = myState.get_position()
        food = self.get_food(state).as_list()
        #print("My agent next position: ", myPosition, " Available food list: ", food)
        if len(food) > 0:  # Should always be true
            return min(self.get_maze_distance(myPosition, dot) for dot in food)
        else:
            return self.gridWidth

    def dist_to_nearest_capsule(self, state):
        # Returns the distance to the nearest capsule

        myState = state.get_agent_state(self.index)
        myPosition = myState.get_position()
        capsules = self.get_capsules(state)
        if len(capsules) > 0:
            return min(self.get_maze_distance(myPosition, item) for item in capsules)
        else:
            return 0

    def dist_to_opponent_ghost(self, successor):
        # Returns distance of the nearest not scared opponent ghost

        myState = successor.get_agent_state(self.index)
        myPosition = myState.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghostList = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None \
                     and a.scared_timer == 0 ]
        if len(ghostList) > 0:
            return min(self.get_maze_distance(myPosition, ghostPos) for ghostPos in ghostList)
        else:
            return self.gridWidth

    def dist_to_invaders(self, successor):
        # Computes distance to invaders we can see
        myState = successor.get_agent_state(self.index)
        myPosition = myState.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:
            dists = [self.get_maze_distance(myPosition, a.get_position()) for a in invaders]
            return min(dists)
        else:
            return self.gridWidth

    def is_myTerritory(self, state):
        myState = state.get_agent_state(self.index)
        myPosition = myState.get_position()
        x, y = myPosition

        if self.red:
            if x < self.gridWidth / 2:
                return True
            else:
                return False
        else:
            if x >= self.gridWidth / 2:
                return True
            else:
                return False

    def count_walls(self, state):
        numWalls = 0
        myState = state.get_agent_state(self.index)
        myPosition = myState.get_position()
        i, j = myPosition
        x = int(i)
        y = int(j)
        if state.has_wall(x+1, y):
            numWalls += 1
        if state.has_wall(x, y+1):
            numWalls += 1
        if state.has_wall(x-1, y):
            numWalls += 1
        if state.has_wall(x, y-1):
            numWalls += 1
        return numWalls

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

class OffensiveQLearningAgent(QLearningCaptureAgent):
    """
  A Qlearning agent that seeks food.
  """

    def get_reward(self, game_state, action):

        #print("Agent is an offensive one with index: ", self.index)
        reward = 0
        if action == 'Stop':
            reward -= self.gridWidth
        successor = self.get_successor(game_state, action)
        myState = successor.get_agent_state(self.index)
        myPosition = myState.get_position()
        distFromStart = self.get_maze_distance(self.start, myPosition)
        foodList = self.get_food(successor).as_list()
        distToFood = self.dist_to_food_to_eat(successor)
        distToCapsule = self.dist_to_nearest_capsule(successor)
        distToGhost = self.dist_to_opponent_ghost(successor)
        currentNumWalls = self.count_walls(game_state)
        nextNumWalls = self.count_walls(successor)
        if self. is_myTerritory(successor):                 # I am on my own team territory
            if myState.is_pacman:                           # I am Pacman --> going back home
                #print("Agent is Pacman on his own territory")
                reward += myState.num_carrying * 10
            else:                                           # I am a ghost trying to reach the opponent territory
                #print("Agent is a ghost on his own territory. NumWalls: ", nextNumWalls, " dist to food: ", distToFood)
                if self.red:
                    reward += distFromStart + myPosition[0]
                else:
                    reward += distFromStart - myPosition[0]
                if nextNumWalls == 0 or nextNumWalls <= currentNumWalls:
                    reward -= min(distToFood, distToCapsule)
                else:
                    if nextNumWalls >= currentNumWalls:
                        reward -= distToFood * nextNumWalls
        else:
            if myState.is_pacman:
                #print("Agent is Pacman on the opponent's territory")
                if len(foodList) <=2:               # Agent must run back home
                    reward = -999
                else:
                    reward += (myState.num_carrying * 3) + distToGhost + (self.gridWidth / 2)
                    if myPosition in foodList:
                        reward += self.gridWidth /2
                    if myPosition in self.get_capsules(successor):
                        reward += self.gridWidth
                    if nextNumWalls == 0:
                        reward -= distToFood
                    else:
                        if nextNumWalls >= currentNumWalls:
                            reward -= distToFood * nextNumWalls
                    if distToGhost < 3:
                        reward -= self.gridWidth * 2
            else:
                reward += distToGhost                       # Shouldn't happen

        return reward


class DefensiveReflexAgent(QLearningCaptureAgent):

    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        #print("Agent is a defensive one with index: ", self.index)
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
