# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # DFS uses Stack with state, path and path_cost.
    frontier = util.Stack()
    initial_state = problem.getStartState()
    frontier.push((initial_state, [], 0))
    # Using the graph search version of DFS
    explored_set = set()
    while not (frontier.isEmpty()):
        current_state, path, path_cost = frontier.pop()
        if problem.isGoalState(current_state):
            return path
        else:
            list_of_successors = problem.getSuccessors(current_state)
            for successor in list_of_successors:
                next_state, action, step_cost = successor
                # If explored skip this state
                if next_state in explored_set:
                    continue
                else:
                    frontier.push((next_state, path + [action], path_cost + step_cost))
                del next_state, action, step_cost, successor  # Free the memory
        # Current state should be marked as explored.
        explored_set.add(current_state)
        del list_of_successors
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # BFS uses Queue with state, path and path_cost.
    frontier = util.Queue()
    initial_state = problem.getStartState()
    frontier.push((initial_state, [], 0))
    current_frontier_states = set()  # To keep track of states present currently in frontier.
    current_frontier_states.add(initial_state)
    explored_set = set()  # To keep track of explored states
    while ~(frontier.isEmpty()):
        current_state, path, path_cost = frontier.pop()
        current_frontier_states.remove(current_state)
        if problem.isGoalState(current_state):
            return path
        else:
            list_of_successors = problem.getSuccessors(current_state)
            for successor in list_of_successors:
                next_state, action, step_cost = successor
                if (next_state in explored_set) or (next_state in current_frontier_states):
                    # Already explored node or already present in the frontier
                    continue
                else:
                    # Successor needs to be added to the frontier
                    frontier.push((next_state, path + [action], path_cost + step_cost))
                    current_frontier_states.add(next_state)
                del next_state, action, step_cost, successor  # Free the memory
        # Add the current state to explored set of states
        explored_set.add(current_state)
    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # UCS uses PriorityQueue with state, path and path_cost and path_cost as priority.
    frontier = util.PriorityQueue()
    initial_state = problem.getStartState()
    frontier.push((initial_state, [], 0), 0)
    current_frontier_states = dict()  # Faster tracking of frontier nodes and the cost to reach the node.
    current_frontier_states[initial_state] = 0
    explored_states = set()  # To keep track of explored states
    while not frontier.isEmpty():
        current_state, path, path_cost = frontier.pop()
        del current_frontier_states[current_state]
        if problem.isGoalState(current_state):
            return path
        else:
            successors = problem.getSuccessors(current_state)
            for successor in successors:
                next_state, action, step_cost = successor
                if next_state in explored_states:
                    continue  # Already explored
                else:
                    # Steps required commonly
                    new_path = path + [action]
                    new_cost = path_cost + step_cost
                    if next_state in current_frontier_states:
                        if new_cost < current_frontier_states[next_state]:  # Frontier has a higher cost --> Update
                            frontier.update((next_state, new_path, new_cost), new_cost)
                            current_frontier_states[next_state] = new_cost
                        else:  # Frontier has low cost already
                            continue
                    else:  # Unexplored state/node
                        frontier.push((next_state, new_path, new_cost), new_cost)
                        current_frontier_states[next_state] = new_cost
        # Add the current state to explored set of states
        explored_states.add(current_state)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # aStar uses PriorityQueue with state, path and path_cost & path_cost + heuristics as priority.
    frontier = util.PriorityQueue()
    initial_state = problem.getStartState()
    path_cost = 0
    current_heuristic = heuristic(initial_state, problem=problem)
    frontier.push((initial_state, [], path_cost), path_cost + current_heuristic)
    frontier_current_states = dict()  # Faster tracking of frontier nodes and the cost to reach the node.
    frontier_current_states[initial_state] = 0
    explored_states = set()  # To keep track of explored states
    while not frontier.isEmpty():
        current_state, path, path_cost = frontier.pop()
        if problem.isGoalState(current_state):
            return path
        else:
            successors = problem.getSuccessors(current_state)
            for successor in successors:
                next_state, action, step_cost = successor
                if next_state in explored_states:
                    continue  # Already explored
                else:
                    # Commonly required steps
                    new_path = path + [action]
                    new_cost = path_cost + step_cost
                    new_cost_plus_heuristics = new_cost + heuristic(next_state, problem=problem)
                    if next_state in frontier_current_states:
                        if new_cost < frontier_current_states[next_state]:  # Frontier has a higher cost --> Update
                            frontier_current_states[next_state] = new_cost
                            frontier.update((next_state, new_path, new_cost), new_cost_plus_heuristics)
                    else:  # Unexplored state/node
                        frontier.push((next_state, new_path, new_cost), new_cost_plus_heuristics)
                        frontier_current_states[next_state] = new_cost
            # Add the current state to explored set of states
            explored_states.add(current_state)
    return []
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
