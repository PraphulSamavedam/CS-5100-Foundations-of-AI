# multiAgents.py
# --------------
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
import math
import random

import util
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """" Idea is to compare the current position and the new position for
        -- Is the next state a Win? -- by pass all the below conditions and 
        1. Reduction in cost for collecting food
        2. Reduction in cost for collecting capsules
        3. Increase in score
        4. Maintaining safe distance from ghosts if un scared.
        5. Try to maintain the direction instead of steering sideways a lot.
        """
        '''' 
        print(f"newPosition:{newPos}\nnewFood:{newFood.asList()}\nnewGhostStates:{newGhostStates}
        \nnewScaredTimes:{newScaredTimes}")
        '''
        from game import Directions
        import math

        if successorGameState.isWin():
            return math.inf

        # By taking this action are we consuming food or getting closer to the food?
        action_aids_in_food = False  # Default-ly assume this action does not aid in food consumption.
        curr_position = currentGameState.getPacmanPosition()
        curr_food_positions = currentGameState.getFood().asList()
        new_food_positions = newFood.asList()
        # curr_food_positions cannot be empty like curr_capsules_positions as empty food locations mean game was won.
        if len(new_food_positions) < len(curr_food_positions):
            #  We have consumed food -- so a good move
            action_aids_in_food = True
        else:
            foods_distances = []
            for food in curr_food_positions:
                foods_distances.append((util.manhattanDistance(food, curr_position), food))
            curr_nearest_food_distance, curr_nearest_food_location = min(foods_distances)
            new_nearest_food_distance = util.manhattanDistance(curr_nearest_food_location, newPos)
            if new_nearest_food_distance < curr_nearest_food_distance:
                action_aids_in_food = True

        # By taking this action are we consuming capsule or getting closer to the capsule?
        action_aids_in_capsule = False  # Default-ly assume this action does not aid in food consumption.
        curr_capsules_positions = currentGameState.getCapsules()
        if curr_capsules_positions != list():
            new_capsules_positions = successorGameState.getCapsules()
            capsules_distances = []
            for capsule in curr_capsules_positions:
                capsules_distances.append((util.manhattanDistance(capsule, curr_position), capsule))
            curr_nearest_capsule_distance, curr_nearest_capsule_location = min(capsules_distances)
            new_nearest_capsule_distance = util.manhattanDistance(curr_nearest_capsule_location, newPos)
            if len(curr_capsules_positions) < len(new_capsules_positions):
                #  We have consumed capsule -- so a good move
                action_aids_in_capsule = True
            elif new_nearest_capsule_distance < curr_nearest_capsule_distance:
                action_aids_in_capsule = True

        # Are we at safe distance from the nearest ghost?
        minGhostDistance = min(
            [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])
        action_is_fatal = False
        if minGhostDistance <= 1:
            action_is_fatal = True

        # Does the action increase the score?
        action_aids_in_score = True if successorGameState.getScore() - currentGameState.getScore() > 0 else False

        # Keep direction to avoid meaningless random movements when upon criteria are not satisfied
        action_retains_direction = True if currentGameState.getPacmanState().getDirection() == action else False

        score = 0
        # Scoring Logic formula
        if action_is_fatal or action == Directions.STOP:
            return score
        score += 1
        if action_retains_direction:
            score += 1
        if action_aids_in_food:
            score += 1
        if action_aids_in_capsule:
            score += 1
        if action_aids_in_score:
            score += 1
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        debug: bool = False
        import math

        def min_value(state: GameState, agent_index: int, depth: int) -> int:
            """This method can be called by any agent"""
            if debug:
                print("In the min function\nCurrent depth:", depth, "Max:Depth", self.depth)
            legal_actions = state.getLegalActions(agentIndex=agent_index)

            if not legal_actions:  # Terminal test
                return self.evaluationFunction(state)

            ghost_agents_cnt = state.getNumAgents() - 1
            utility = math.inf
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                if agent_index == ghost_agents_cnt:  # Next move is by Pacman
                    utility = min(utility, max_value(successor_state, 0, depth))
                else:  # Next move is by Ghost
                    utility = min(utility, min_value(successor_state, agent_index + 1, depth))
            return utility

        def max_value(state: gameState, agentIndex: int, depth: int) -> int:
            if debug:
                print("In the Max function:\nCurrent depth:", depth, "Max:Depth", self.depth)
            if agentIndex != 0:
                raise BaseException({"String": "Max_value cannot be called by ghost agent"})

            legal_actions = state.getLegalActions(0)

            if not legal_actions or depth == self.depth:
                # Either we have reached the depth or out of moves in the state
                return self.evaluationFunction(state)

            utility = -math.inf
            for action in legal_actions:
                successorState = state.generateSuccessor(0, action)
                utility = max(utility, min_value(successorState, 0 + 1, depth + 1))
            return utility

        pacman_actions_values = []
        pacman_legal_actions = gameState.getLegalActions(0)
        for pacman_action in pacman_legal_actions:
            successorState = gameState.generateSuccessor(0, pacman_action)
            pacman_actions_values.append((min_value(successorState, agent_index=1, depth=1), pacman_action))
        max_value, pacman_action_for_max_value = max(pacman_actions_values)
        return pacman_action_for_max_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        debug: bool = False
        import math
        utility_dict = util.Counter()
        ghost_agents_count = gameState.getNumAgents() - 1
        alpha = -math.inf
        beta = math.inf

        def max_value(state, agent, depth, alpha, beta):
            utility = -math.inf  # initial utility of max node
            legal_actions = state.getLegalActions(agent)
            if depth == self.depth or not legal_actions:
                # reached max depth -> evaluate utility of state
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                min_v = min_value(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta)
                utility = max(utility, min_v)
                if utility > beta:
                    return utility
                alpha = max(alpha, utility)
            return utility

        def min_value(state, agent, depth, alpha, beta):
            utility = math.inf  # initial utility of min node
            legal_actions = state.getLegalActions(agent)
            if not legal_actions:
                return self.evaluationFunction(state)
            if agent == ghost_agents_count:
                for action in state.getLegalActions(agent):
                    max_v = max_value(state.generateSuccessor(agent, action), 0, depth + 1, alpha, beta)
                    utility = min(utility, max_v)
                    if utility < alpha:
                        return utility
                    beta = min(beta, utility)
            else:
                # min <- min agent (idx: agent+1) of same layer
                for action in state.getLegalActions(agent):
                    min_v = min_value(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta)
                    utility = min(utility, min_v)
                    if utility < alpha:
                        return utility
                    beta = min(beta, utility)
            return utility

        for action in gameState.getLegalActions(0):
            value = min_value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            utility_dict[action] = value
            alpha = max(alpha, value)
        return utility_dict.argMax()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        utility_dict = util.Counter()
        ghost_agents_count = gameState.getNumAgents() - 1  # last min agent in ply

        def exp_value(state, agent, depth):
            value = 0
            legal_actions = state.getLegalActions(agent)
            if not legal_actions:
                return self.evaluationFunction(state)
            else:
                probability = 1.0 / len(legal_actions)
                if agent == ghost_agents_count:
                    # Next move needs to be done by Pacman in next depth
                    for action in legal_actions:
                        max_v = max_value(state.generateSuccessor(agent, action), 0, depth + 1)
                        value += probability * max_v  # expectation
                else:
                    # Next exp value is from ghost of same depth
                    for action in legal_actions:
                        exp_v = exp_value(state.generateSuccessor(agent, action), agent + 1, depth)
                        value += probability * exp_v  # expectation
                return value

        def max_value(state, agent, depth):
            value = -math.inf  # initial utility_dict of max node
            legal_actions = state.getLegalActions(agent)
            if depth == self.depth or not legal_actions:
                # reached max depth -> evaluate utility_dict of state
                return self.evaluationFunction(state)
            for action in state.getLegalActions(agent):
                exp_v = exp_value(state.generateSuccessor(agent, action), agent + 1, depth)
                value = max(value, exp_v)
            return value

        for action in gameState.getLegalActions(0):
            utility_dict[action] = exp_value(gameState.generateSuccessor(0, action), 1, 0)
        return utility_dict.argMax()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    current_position: (int, int) = currentGameState.getPacmanPosition()
    current_food: list = currentGameState.getFood().asList()
    current_ghost_states = currentGameState.getGhostStates()
    current_capsules: list = currentGameState.getCapsules()
    current_score: float = currentGameState.getScore()
    currrent_scared_times: int = sum([ghostState.scaredTimer for ghostState in current_ghost_states])
    walls: list = currentGameState.getWalls().asList()

    def total_walls_between(pos, food):
        x, y = pos
        fx, fy = food
        fx, fy = int(fx), int(fy)

        return sum([wx in range(min(x, fx), max(x, fx) + 1) and
                    wy in range(min(y, fy), max(y, fy) + 1) for (wx, wy) in walls])

    foodDistance = [util.manhattanDistance(current_position, food) for food in current_food]

    if currentGameState.isWin():
        score = 10000
    else:
        sorted_food_distance: list = sorted(foodDistance)
        close_food_distance: int = sum(sorted_food_distance[-3:])
        nearest_food_distance: int = sum(sorted_food_distance[-1:])
        ghost_distance = [util.manhattanDistance(current_position, ghost.getPosition()) +
                          2 * total_walls_between(current_position, ghost.getPosition())
                          for ghost in current_ghost_states]
        min_ghost_distance: int = min(min(ghost_distance), 4)  # To ensure I do not have to wait till Ghost is near.

        score = current_score + currrent_scared_times + min_ghost_distance - len(current_capsules)
        score += 1.0 / len(current_food) + (1.0 / close_food_distance) + (1.0 / nearest_food_distance)
    return score


# Abbreviation
better = betterEvaluationFunction
