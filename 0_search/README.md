# Search Algorithms Project

## Overview

This project implements **classic search algorithms** for the Pacman game. The system explores uninformed and informed search strategies including Depth-First Search (DFS), Breadth-First Search (BFS), Uniform Cost Search (UCS), and A* Search. Additionally, it tackles complex search problems like finding optimal paths through multiple goal states and designing effective heuristics.

The implementation demonstrates how search algorithms can solve navigation and planning problems in maze environments, showing the trade-offs between optimality, completeness, and computational efficiency.

## Search Algorithms Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEARCH ALGORITHMS SYSTEM                                │
│                        System Workflow Diagram                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │    SEARCH PROBLEM                │
                    │  - Initial State                 │
                    │  - Goal Test                     │
                    │  - Successor Function            │
                    │  - Cost Function                 │
                    └──────────────┬───────────────────┘
                                   │
        ┌──────────────────────────┼────────────────────────────┐
        │                          │                            │
        ▼                          ▼                            ▼
┌────────────────┐      ┌────────────────┐        ┌────────────────┐
│ UNINFORMED     │      │ INFORMED       │        │ HEURISTIC      │
│ SEARCH         │      │ SEARCH         │        │ FUNCTIONS      │
│                │      │                │        │                │
│ - DFS          │      │ - Greedy       │        │ - Manhattan    │
│ - BFS          │      │ - A* Search    │        │   Distance     │
│ - UCS          │      │                │        │ - Euclidean    │
└────────┬───────┘      └────────┬───────┘        │ - Custom       │
         │                       │                 └────────┬───────┘
         └───────────────────────┼─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FRONTIER / FRINGE     │
                    │  - Stack (DFS)          │
                    │  - Queue (BFS)          │
                    │  - Priority Queue (UCS) │
                    │  - Priority Queue (A*)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   EXPLORED SET          │
                    │  - Track visited states │
                    │  - Avoid cycles         │
                    │  - Graph search         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SOLUTION PATH         │
                    │  - Sequence of actions  │
                    │  - Cost                 │
                    │  - Nodes expanded       │
                    └─────────────────────────┘

Key Algorithms:
├─ DFS: Stack-based, explores deep paths first
├─ BFS: Queue-based, explores level by level
├─ UCS: Priority queue by path cost
└─ A*: Priority queue by f(n) = g(n) + h(n)
```

## Features

### Core Capabilities

1. **Depth-First Search (DFS)** - Stack-based deep exploration
2. **Breadth-First Search (BFS)** - Queue-based level-order search
3. **Uniform Cost Search (UCS)** - Optimal path by cost
4. **A* Search** - Informed search with heuristics
5. **Custom Heuristics** - Problem-specific heuristic design
6. **Multiple Goal Problems** - Finding optimal paths through multiple goals

### Search Algorithms

#### 1. Depth-First Search (DFS)
- **Strategy:** Explore deepest nodes first
- **Data Structure:** Stack (LIFO)
- **Complete:** No (can get stuck in infinite paths)
- **Optimal:** No
- **Time:** O(b^m)
- **Space:** O(bm)

#### 2. Breadth-First Search (BFS)
- **Strategy:** Explore shallowest nodes first
- **Data Structure:** Queue (FIFO)
- **Complete:** Yes
- **Optimal:** Yes (if step costs equal)
- **Time:** O(b^d)
- **Space:** O(b^d)

#### 3. Uniform Cost Search (UCS)
- **Strategy:** Expand least-cost node
- **Data Structure:** Priority Queue (by path cost)
- **Complete:** Yes
- **Optimal:** Yes
- **Time:** O(b^(C*/ε))
- **Space:** O(b^(C*/ε))

#### 4. A* Search
- **Strategy:** Expand node with lowest f(n) = g(n) + h(n)
- **Data Structure:** Priority Queue (by f-value)
- **Complete:** Yes (with admissible heuristic)
- **Optimal:** Yes (with admissible heuristic)
- **Time:** Exponential (depends on heuristic)
- **Space:** O(b^d)

### Problem Types

#### Position Search
- **Goal:** Reach a specific location
- **State:** (x, y) coordinates
- **Actions:** North, South, East, West
- **Cost:** 1 per step

#### Corners Problem
- **Goal:** Visit all four corners
- **State:** (position, visited_corners)
- **Actions:** North, South, East, West
- **Cost:** 1 per step

#### Food Search
- **Goal:** Collect all food pellets
- **State:** (position, remaining_food)
- **Actions:** North, South, East, West
- **Cost:** 1 per step

## Quick Start

### Requirements
- **Python 3.6+** - For running search algorithms
- All dependencies included in project files
- No external packages required

### Running Search Algorithms

#### DFS on Medium Maze
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch
```

#### BFS on Medium Maze
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=breadthFirstSearch
```

#### UCS with Different Costs
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=uniformCostSearch
```

#### A* Search
```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Command Line Options

**Parameters:**
- `-l` - Layout/maze (tinyMaze, mediumMaze, bigMaze, etc.)
- `-p` - Agent type (SearchAgent)
- `-a` - Agent arguments (fn=algorithm, heuristic=function)
- `-z` - Zoom level (0.5 = half size)
- `--frameTime=0` - Run at maximum speed

## Usage

### 1. Depth-First Search

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
```

**Characteristics:**
- Explores deep into the maze before backtracking
- May find suboptimal solutions
- Memory efficient
- Fast for deep solutions

**When to Use:**
- Memory is limited
- Solution depth is known
- Any solution is acceptable

### 2. Breadth-First Search

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

**Characteristics:**
- Explores level by level
- Finds shortest path (if costs equal)
- Memory intensive
- Guaranteed optimal (unit costs)

**When to Use:**
- Shortest path needed
- Sufficient memory available
- Solution is shallow

### 3. Uniform Cost Search

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
```

**Characteristics:**
- Expands cheapest node first
- Finds least-cost path
- Handles varying step costs
- More expansions than A*

**When to Use:**
- Variable action costs
- Optimal solution required
- No good heuristic available

### 4. A* Search

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

**Characteristics:**
- Uses heuristic to guide search
- Optimal with admissible heuristic
- Fewer expansions than UCS
- Requires good heuristic

**When to Use:**
- Good heuristic available
- Optimal solution needed
- Want to minimize expansions

### 5. Corners Problem

Find path visiting all four corners:

```bash
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

**With A*:**
```bash
python pacman.py -l mediumCorners -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic
```

### 6. Food Search Problem

Collect all food pellets:

```bash
python pacman.py -l trickySearch -p SearchAgent -a fn=aStarSearch,prob=FoodSearchProblem,heuristic=foodHeuristic
```

## How It Works

### Generic Search Algorithm

```
function GRAPH-SEARCH(problem):
    frontier = data_structure([problem.initial_state])
    explored = set()

    while frontier not empty:
        node = frontier.pop()

        if problem.goal_test(node.state):
            return solution(node)

        explored.add(node.state)

        for action in problem.actions(node.state):
            child = child_node(problem, node, action)

            if child.state not in explored and child not in frontier:
                frontier.push(child)

    return failure
```

### Algorithm Implementations

#### Depth-First Search
```python
def depthFirstSearch(problem):
    frontier = util.Stack()
    explored = set()

    # Push (state, actions, cost)
    frontier.push((problem.getStartState(), [], 0))

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state in explored:
            continue

        explored.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in explored:
                newActions = actions + [action]
                frontier.push((successor, newActions, cost + stepCost))

    return []  # No solution found
```

#### Breadth-First Search
```python
def breadthFirstSearch(problem):
    frontier = util.Queue()
    explored = set()

    frontier.push((problem.getStartState(), [], 0))

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state in explored:
            continue

        explored.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in explored:
                newActions = actions + [action]
                frontier.push((successor, newActions, cost + stepCost))

    return []
```

#### Uniform Cost Search
```python
def uniformCostSearch(problem):
    frontier = util.PriorityQueue()
    explored = set()

    # Priority = path cost
    frontier.push((problem.getStartState(), [], 0), 0)

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state in explored:
            continue

        explored.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in explored:
                newActions = actions + [action]
                newCost = cost + stepCost
                frontier.push((successor, newActions, newCost), newCost)

    return []
```

#### A* Search
```python
def aStarSearch(problem, heuristic=nullHeuristic):
    frontier = util.PriorityQueue()
    explored = set()

    # Priority = g(n) + h(n)
    startState = problem.getStartState()
    frontier.push((startState, [], 0), heuristic(startState, problem))

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state in explored:
            continue

        explored.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in explored:
                newActions = actions + [action]
                newCost = cost + stepCost
                # f(n) = g(n) + h(n)
                priority = newCost + heuristic(successor, problem)
                frontier.push((successor, newActions, newCost), priority)

    return []
```

### Heuristic Functions

#### Manhattan Distance
```python
def manhattanHeuristic(position, problem, info={}):
    """Manhattan distance to closest goal"""
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
```

#### Corners Heuristic
```python
def cornersHeuristic(state, problem):
    """
    Estimate cost to visit all unvisited corners
    """
    position, visitedCorners = state
    corners = problem.corners

    # Find unvisited corners
    unvisited = [c for c in corners if c not in visitedCorners]

    if not unvisited:
        return 0

    # Use max distance to any unvisited corner
    distances = [util.manhattanDistance(position, corner) for corner in unvisited]
    return max(distances)
```

#### Food Heuristic
```python
def foodHeuristic(state, problem):
    """
    Estimate cost to collect all remaining food
    """
    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Maximum distance to any food
    maxDist = max([util.manhattanDistance(position, food) for food in foodList])

    # Consider food spread
    if len(foodList) > 1:
        # Approximate by farthest food + half of maze
        mazeSize = foodGrid.width + foodGrid.height
        return maxDist + mazeSize // 4

    return maxDist
```

## Performance Comparison

### Small Maze
| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS       | 10        | 15             | 0.01s|
| BFS       | 10        | 15             | 0.01s|
| UCS       | 10        | 15             | 0.01s|
| A*        | 10        | 14             | 0.01s|

### Medium Maze
| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS       | 130       | 146            | 0.02s|
| BFS       | 68        | 269            | 0.03s|
| UCS       | 68        | 269            | 0.03s|
| A*        | 68        | 221            | 0.02s|

### Big Maze
| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS       | 210       | 390            | 0.05s|
| BFS       | 210       | 620            | 0.10s|
| UCS       | 210       | 620            | 0.10s|
| A*        | 210       | 549            | 0.08s|

**Observations:**
- DFS is fast but finds suboptimal paths
- BFS/UCS guarantee optimal but expand many nodes
- A* with good heuristic expands fewest nodes
- A* performance depends on heuristic quality

## Project Structure

```
search/
├── search.py              # Search algorithm implementations
│   ├── depthFirstSearch
│   ├── breadthFirstSearch
│   ├── uniformCostSearch
│   └── aStarSearch
│
├── searchAgents.py        # Search problem definitions
│   ├── PositionSearchProblem
│   ├── CornersProblem
│   ├── FoodSearchProblem
│   └── Heuristics
│
├── pacman.py              # Main game engine
├── game.py                # Game mechanics
├── util.py                # Data structures
│   ├── Stack
│   ├── Queue
│   └── PriorityQueue
│
├── layouts/               # Maze files
│   ├── tinyMaze.lay
│   ├── mediumMaze.lay
│   ├── bigMaze.lay
│   └── openMaze.lay
│
├── autograder.py          # Testing framework
└── test_cases/            # Test scenarios
```

## Testing

```bash
# All tests
python autograder.py

# Individual questions
python autograder.py -q q1  # DFS
python autograder.py -q q2  # BFS
python autograder.py -q q3  # UCS
python autograder.py -q q4  # A*
python autograder.py -q q5  # Corners Problem
python autograder.py -q q6  # Corners Heuristic
python autograder.py -q q7  # Food Heuristic
python autograder.py -q q8  # Suboptimal Search
```

## License

This project is part of academic coursework for CS-5100 at Northeastern University.

**Attribution:** The Pacman AI projects were developed at UC Berkeley by John DeNero and Dan Klein.

### 📚 Usage as Reference

This repository is intended as a **learning resource and reference guide**. Please respect academic integrity policies at your institution.

## Acknowledgments

- **UC Berkeley CS188** for the Pacman AI framework
- **CS-5100 course staff** for guidance and project specifications
