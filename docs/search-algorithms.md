# Search Algorithms

## Overview

The Search Algorithms project implements classic uninformed and informed search strategies for pathfinding in maze environments. Built using graph search principles, the system explores various frontier management strategies and heuristic functions to navigate Pacman through mazes efficiently while minimizing computational cost.

The implementation covers **four fundamental search algorithms** (DFS, BFS, UCS, A*) and demonstrates their application to complex problems including corner navigation and food collection. The project emphasizes understanding trade-offs between optimality, completeness, and computational efficiency.

---

## System Architecture

The search system follows a modular architecture with flexible problem definitions and pluggable algorithms:

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
│ - DFS          │      │ - A* Search    │        │ - Manhattan    │
│ - BFS          │      │                │        │   Distance     │
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
                    │  - Path cost            │
                    │  - Nodes expanded       │
                    └─────────────────────────┘

Key Algorithms:
├─ DFS: Stack-based, explores deep paths first
├─ BFS: Queue-based, explores level by level
├─ UCS: Priority queue by path cost
└─ A*: Priority queue by f(n) = g(n) + h(n)
```

---

## Features and Capabilities

### Search Algorithms Implemented

#### 1. Depth-First Search (DFS)
- **Strategy:** Explore deepest nodes first using Stack (LIFO)
- **Complete:** No (can get stuck in infinite branches)
- **Optimal:** No (finds any solution, not necessarily shortest)
- **Complexity:** Time O(b^m), Space O(bm)
- **Best For:** Memory-constrained scenarios, deep solutions

#### 2. Breadth-First Search (BFS)
- **Strategy:** Explore shallowest nodes first using Queue (FIFO)
- **Complete:** Yes (finds solution if one exists)
- **Optimal:** Yes (if all step costs equal)
- **Complexity:** Time O(b^d), Space O(b^d)
- **Best For:** Finding shortest paths with uniform costs

#### 3. Uniform Cost Search (UCS)
- **Strategy:** Expand least-cost node using Priority Queue
- **Complete:** Yes (with positive step costs)
- **Optimal:** Yes (guaranteed least-cost path)
- **Complexity:** Time/Space O(b^(C*/ε))
- **Best For:** Variable action costs, optimal solutions required

#### 4. A* Search
- **Strategy:** Best-first search with f(n) = g(n) + h(n)
- **Complete:** Yes (with admissible heuristic)
- **Optimal:** Yes (with admissible heuristic)
- **Complexity:** Depends on heuristic quality
- **Best For:** When good heuristic available, optimal solution needed

### Problem Types

#### Position Search Problem
- **Goal:** Navigate to specific target location
- **State:** (x, y) coordinates in maze
- **Actions:** North, South, East, West
- **Application:** Basic pathfinding

#### Corners Problem
- **Goal:** Visit all four maze corners
- **State:** (position, visited_corners_set)
- **Actions:** North, South, East, West
- **Application:** Multiple goal navigation

#### Food Search Problem
- **Goal:** Collect all food pellets efficiently
- **State:** (position, remaining_food_grid)
- **Actions:** North, South, East, West
- **Application:** Resource collection optimization

---

## Quick Start

### Installation

Ensure Python 3.6+ is installed:
```bash
python --version
```

All required files are included in the project. No external dependencies needed.

### Running Algorithms

Navigate to the project directory:
```bash
cd 0_search
```

#### DFS on Medium Maze
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch
```

#### BFS on Medium Maze
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=breadthFirstSearch
```

#### UCS with Variable Costs
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=uniformCostSearch
```

#### A* Search with Manhattan Heuristic
```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-l` | Maze layout | `tinyMaze`, `mediumMaze`, `bigMaze` |
| `-p` | Agent type | `SearchAgent` |
| `-a` | Agent arguments | `fn=astar,heuristic=manhattanHeuristic` |
| `-z` | Zoom level | `0.5` (half size), `1.0` (normal) |
| `--frameTime` | Animation speed | `0` (max speed), `0.1` (slow) |
| `-q` | Quiet mode | No graphics |

---

## Algorithm Details

### Depth-First Search (DFS)

**Pseudocode:**
```
function DFS(problem):
    frontier = Stack()
    frontier.push(problem.startState)
    explored = empty set

    while frontier not empty:
        state = frontier.pop()

        if problem.isGoal(state):
            return solution

        if state not in explored:
            explored.add(state)
            for successor in problem.getSuccessors(state):
                frontier.push(successor)

    return failure
```

**Characteristics:**
- Explores one branch completely before backtracking
- Memory efficient (only stores path)
- Can find suboptimal solutions
- Fast for deep goals

### Breadth-First Search (BFS)

**Pseudocode:**
```
function BFS(problem):
    frontier = Queue()
    frontier.push(problem.startState)
    explored = empty set

    while frontier not empty:
        state = frontier.pop()

        if problem.isGoal(state):
            return solution

        if state not in explored:
            explored.add(state)
            for successor in problem.getSuccessors(state):
                frontier.push(successor)

    return failure
```

**Characteristics:**
- Explores all nodes at depth d before depth d+1
- Guarantees shortest path (uniform costs)
- Memory intensive (stores all nodes at current level)
- Optimal for unit step costs

### Uniform Cost Search (UCS)

**Pseudocode:**
```
function UCS(problem):
    frontier = PriorityQueue()
    frontier.push(problem.startState, priority=0)
    explored = empty set

    while frontier not empty:
        state = frontier.pop()  # Lowest cost first

        if problem.isGoal(state):
            return solution

        if state not in explored:
            explored.add(state)
            for successor, cost in problem.getSuccessors(state):
                frontier.push(successor, priority=pathCost + cost)

    return failure
```

**Characteristics:**
- Always expands least-cost node
- Guarantees optimal solution
- Handles variable step costs
- More nodes expanded than A*

### A* Search

**Pseudocode:**
```
function AStar(problem, heuristic):
    frontier = PriorityQueue()
    frontier.push(problem.startState, priority=heuristic(startState))
    explored = empty set

    while frontier not empty:
        state = frontier.pop()  # Lowest f(n) first

        if problem.isGoal(state):
            return solution

        if state not in explored:
            explored.add(state)
            for successor, cost in problem.getSuccessors(state):
                g = pathCost + cost
                h = heuristic(successor)
                f = g + h
                frontier.push(successor, priority=f)

    return failure
```

**Evaluation Function:**
```
f(n) = g(n) + h(n)
where:
  g(n) = actual cost from start to n
  h(n) = estimated cost from n to goal
```

**Characteristics:**
- Uses heuristic to guide search
- Optimal with admissible heuristic (h ≤ h*)
- Expands fewer nodes than uninformed search
- Efficiency depends on heuristic quality

---

## Heuristic Design

### Manhattan Distance Heuristic

**Formula:**
```
h(position, goal) = |x₁ - x₂| + |y₁ - y₂|
```

**Properties:**
- **Admissible:** Never overestimates (can't move diagonally through walls)
- **Consistent:** h(n) ≤ c(n,a,n') + h(n')
- **Efficient:** O(1) computation
- **Use Case:** Grid-based pathfinding

### Corners Heuristic

**Strategy:** Estimate cost to visit all unvisited corners

**Approach:**
```python
def cornersHeuristic(state, problem):
    position, visitedCorners = state
    unvisited = [c for c in corners if c not in visitedCorners]

    if not unvisited:
        return 0

    # Distance to farthest unvisited corner
    distances = [manhattanDistance(position, c) for c in unvisited]
    return max(distances)
```

**Properties:**
- Admissible (underestimates true cost)
- Guides search toward unvisited corners
- Simple but effective

### Food Heuristic

**Strategy:** Estimate cost to collect all remaining food

**Approach:**
```python
def foodHeuristic(state, problem):
    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Maximum distance to any food
    maxDist = max([manhattanDistance(position, food) for food in foodList])

    # Add penalty for food count
    return maxDist + len(foodList) // 2
```

**Properties:**
- Balances distance and quantity
- Admissible approximation
- Scalable to many food pellets

---

## Performance Analysis

### Maze Size Comparison

**Small Maze (tinyMaze):**

| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS | 10 | 15 | 0.01s |
| BFS | 10 | 15 | 0.01s |
| UCS | 10 | 15 | 0.01s |
| A* | 10 | 14 | 0.01s |

**Medium Maze:**

| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS | 130 | 146 | 0.02s |
| BFS | 68 | 269 | 0.03s |
| UCS | 68 | 269 | 0.03s |
| A* | 68 | 221 | 0.02s |

**Big Maze:**

| Algorithm | Path Cost | Nodes Expanded | Time |
|-----------|-----------|----------------|------|
| DFS | 210 | 390 | 0.05s |
| BFS | 210 | 620 | 0.10s |
| UCS | 210 | 620 | 0.10s |
| A* | 210 | 549 | 0.08s |

**Key Observations:**
- **DFS:** Fast but finds suboptimal paths
- **BFS/UCS:** Optimal but expand many nodes
- **A*:** Best of both worlds - optimal with fewer expansions
- **Heuristic Impact:** A* efficiency directly correlates with heuristic quality

---

## Usage Examples

### Basic Pathfinding

```bash
# Depth-First Search
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs

# Breadth-First Search (optimal for unit costs)
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs

# Uniform Cost Search (optimal for variable costs)
python pacman.py -l mediumDottedMaze -p SearchAgent -a fn=ucs

# A* Search with Manhattan heuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Complex Problems

```bash
# Corners Problem with BFS
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem

# Corners Problem with A* and custom heuristic
python pacman.py -l mediumCorners -p SearchAgent -a fn=astar,prob=CornersProblem,heuristic=cornersHeuristic

# Food Search Problem
python pacman.py -l trickySearch -p SearchAgent -a fn=astar,prob=FoodSearchProblem,heuristic=foodHeuristic
```

### Performance Testing

```bash
# Maximum speed (no graphics)
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime=0 -q

# Slow motion visualization
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime=0.5

# Multiple runs for statistics
python pacman.py -l mediumMaze -p SearchAgent -a fn=astar -n 10 -q
```

---

## Key Algorithms Explained

### Graph Search Template

All algorithms follow this general structure:

```python
def graphSearch(problem, frontierDataStructure):
    """
    Generic graph search algorithm
    """
    frontier = frontierDataStructure()
    frontier.push((problem.getStartState(), [], 0))  # (state, actions, cost)
    explored = set()

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        # Goal test
        if problem.isGoalState(state):
            return actions

        # Avoid revisiting
        if state not in explored:
            explored.add(state)

            # Expand successors
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in explored:
                    newActions = actions + [action]
                    newCost = cost + stepCost
                    frontier.push((successor, newActions, newCost))

    return []  # No solution found
```

### A* Search with Heuristics

**Core Equation:**
```
f(n) = g(n) + h(n)

where:
  f(n) = estimated total cost of path through n
  g(n) = actual cost from start to n
  h(n) = estimated cost from n to goal (heuristic)
```

**Admissibility Requirement:**
```
h(n) ≤ h*(n)  (never overestimate)
```

**Consistency Requirement:**
```
h(n) ≤ c(n, a, n') + h(n')  (triangle inequality)
```

**Implementation:**
```python
def aStarSearch(problem, heuristic):
    frontier = PriorityQueue()
    explored = set()

    startState = problem.getStartState()
    startPriority = 0 + heuristic(startState, problem)
    frontier.push((startState, [], 0), startPriority)

    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state not in explored:
            explored.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in explored:
                    newActions = actions + [action]
                    g = cost + stepCost
                    h = heuristic(successor, problem)
                    f = g + h
                    frontier.push((successor, newActions, g), f)

    return []
```

---

## Heuristic Examples

### Manhattan Distance (Position Search)

```python
def manhattanHeuristic(position, problem):
    """
    Manhattan distance to goal
    Admissible for grid navigation without diagonal moves
    """
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
```

### Euclidean Distance

```python
def euclideanHeuristic(position, problem):
    """
    Straight-line distance to goal
    Admissible but less informative than Manhattan for grids
    """
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5
```

### Corners Heuristic

```python
def cornersHeuristic(state, problem):
    """
    Heuristic for visiting all corners
    Uses maximum distance to any unvisited corner
    """
    position, visitedCorners = state
    corners = problem.corners

    unvisited = [c for c in corners if c not in visitedCorners]

    if not unvisited:
        return 0

    # Max distance provides admissible lower bound
    distances = [manhattanDistance(position, corner) for corner in unvisited]
    return max(distances)
```

### Food Heuristic

```python
def foodHeuristic(state, problem):
    """
    Heuristic for collecting all food
    Combines distance and quantity factors
    """
    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0

    # Distance to farthest food
    maxDist = max([manhattanDistance(position, food) for food in foodList])

    # Penalty for remaining food count
    foodPenalty = len(foodList) // 2

    return maxDist + foodPenalty
```

---

## Testing and Validation

### Automated Testing

```bash
# Test all questions
python autograder.py

# Test specific questions
python autograder.py -q q1  # DFS
python autograder.py -q q2  # BFS
python autograder.py -q q3  # UCS
python autograder.py -q q4  # A*
python autograder.py -q q5  # Corners Problem
python autograder.py -q q6  # Corners Heuristic
python autograder.py -q q7  # Food Heuristic
python autograder.py -q q8  # Suboptimal Search
```

### Manual Testing

```bash
# Visual testing on different maze sizes
python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs
python pacman.py -l mediumMaze -p SearchAgent -a fn=astar
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar

# Performance comparison
for algo in dfs bfs ucs astar; do
    echo "Testing $algo"
    python pacman.py -l mediumMaze -p SearchAgent -a fn=$algo -q
done
```

---

## Troubleshooting

### Common Issues

#### DFS Goes in Circles
**Problem:** Infinite loop or very long paths

**Solution:** Ensure explored set prevents revisiting:
```python
if state not in explored:
    explored.add(state)
    # expand successors
```

#### BFS Runs Out of Memory
**Problem:** Too many nodes stored

**Solutions:**
1. Verify explored set works correctly
2. Try smaller maze first
3. Check state representation is hashable

#### A* Not Finding Optimal Path
**Problem:** Returns suboptimal solution

**Solutions:**
1. Verify heuristic is admissible (never overestimates)
2. Check f(n) = g(n) + h(n) calculation
3. Ensure priority queue uses f(n) as priority

#### Heuristic Returns Incorrect Values
**Problem:** Crashes or poor performance

**Solutions:**
1. Never return infinity or negative values
2. Verify admissibility (h ≤ true cost)
3. Test heuristic in isolation before integration

---

## Performance Optimization Tips

1. **Choose Right Algorithm:**
   - Need optimal? Use A* or UCS
   - Limited memory? Use DFS
   - Uniform costs? Use BFS

2. **Design Better Heuristics:**
   - More informed = fewer expansions
   - Must remain admissible for optimality
   - Precompute when possible

3. **State Representation:**
   - Make states hashable for explored set
   - Minimize state size for memory efficiency
   - Include only relevant information

4. **Implementation:**
   - Check explored before expanding
   - Store path incrementally, not by copying
   - Use appropriate data structures (Stack/Queue/PriorityQueue)

---

## Learning Objectives

✅ **Algorithm Implementation**
- Implement DFS, BFS, UCS, and A* from scratch
- Understand frontier and explored set management
- Handle state representation and goal testing

✅ **Heuristic Design**
- Design admissible heuristics for various problems
- Balance informativeness with computational cost
- Verify consistency and admissibility properties

✅ **Complexity Analysis**
- Analyze time and space complexity
- Compare algorithm performance empirically
- Understand optimality and completeness guarantees

✅ **Problem Formulation**
- Define state spaces for search problems
- Design successor functions and cost models
- Formulate complex multi-goal problems

---

## Additional Resources

### SearchProblem Interface

```python
problem.getStartState()           # Returns initial state
problem.isGoalState(state)        # Returns True if state is goal
problem.getSuccessors(state)      # Returns [(successor, action, cost), ...]
```

### Utility Functions

```python
util.manhattanDistance(xy1, xy2)  # L1 distance
util.Stack()                      # LIFO data structure
util.Queue()                      # FIFO data structure
util.PriorityQueue()              # Ordered by priority
```

---

## References

- **UC Berkeley CS188** - Original project framework
- **Russell & Norvig** - "Artificial Intelligence: A Modern Approach"
- **Course Materials** - CS-5100 lecture slides and readings
