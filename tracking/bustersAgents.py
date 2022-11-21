# bustersAgents.py
# ----------------
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


import busters
import inference
import util
from keyboardAgents import KeyboardAgent
from util import raiseNotDefined


class NullGraphics:
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        noisyDistance = observation
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if noisyDistance != None and \
                    busters.getObservationProbability(noisyDistance, trueDistance) > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        try:
            inferenceType = util.lookup(inference, globals())
        except Exception:
            inferenceType = util.lookup('inference.' + inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.observeEnable:
                inf.observe(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)


from distanceCalculator import Distancer
from game import Directions


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState: busters.GameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ########### ########### ###########
    ########### QUESTION 8  ###########
    ########### ########### ###########

    def chooseAction(self, gameState: busters.GameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closest to the closest ghost (according to mazeDistance!).
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        "*** YOUR CODE HERE ***"
        import math
        from game import Actions
        # Find the most likely position of the living ghost.
        likely_positions = list()
        for live_ghost_distro in livingGhostPositionDistributions:
            max_prob = -1
            likely_position = None
            for position in live_ghost_distro.keys():
                if live_ghost_distro[position] > max_prob:
                    max_prob = live_ghost_distro[position]
                    likely_position = position
            likely_positions.append(likely_position)
        # Find the nearest ghost among the living ghost.
        min_distance = math.inf
        closest_ghost = None
        for position in likely_positions:
            distance = self.distancer.getDistance(pacmanPosition, position)
            if min_distance > distance:
                min_distance = distance
                closest_ghost = position
        # Find the action which minimizes the closest ghost distance.
        min_value = math.inf
        best_action = None
        for action in legal:
            new_position = Actions.getSuccessor(pacmanPosition, action)
            action_value = self.distancer.getDistance(new_position, closest_ghost)
            if min_value > action_value:
                min_value = action_value
                best_action = action
        return best_action

        raiseNotDefined()

        "*** END YOUR CODE HERE ***"
