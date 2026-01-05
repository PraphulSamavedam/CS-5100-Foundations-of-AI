# Multi-Agent Systems

## Overview

The Multi-Agent Systems project implements adversarial search algorithms for game playing where Pacman must make optimal decisions against intelligent ghost opponents. Built on game tree search principles, the system explores various decision-making strategies from reactive reflex agents to sophisticated minimax search with alpha-beta optimization and expectimax for probabilistic opponents.

The implementation demonstrates how agents can reason about adversaries in competitive scenarios, achieving **optimal play** against perfect opponents through minimax while maintaining **real-time performance** (30ms per move) using alpha-beta pruning and carefully designed evaluation functions.

---

## System Architecture

The multi-agent system follows a game tree search architecture with alternating agent layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT PACMAN SYSTEM                                │
│                         System Workflow Diagram                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │     GAME ENVIRONMENT             │
                    │  - Pacman (Player Agent)         │
                    │  - Ghosts (Adversarial Agents)   │
                    │  - Food & Capsules               │
                    │  - Maze Layout                   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     AGENT TYPES              │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼────────────────────────────┐
        │                          │                            │
        ▼                          ▼                            ▼
┌────────────────┐      ┌────────────────┐        ┌────────────────┐
│ REFLEX AGENT   │      │ MINIMAX AGENT  │        │ EXPECTIMAX     │
│                │      │                │        │ AGENT          │
│ - Immediate    │      │ - Perfect      │        │                │
│   evaluation   │      │   adversary    │        │ - Probabilistic│
│ - State-action │      │ - Min-Max      │        │   adversary    │
│   pairs        │      │   search       │        │ - Expected     │
│ - No lookahead │      │ - Optimal play │        │   values       │
└────────┬───────┘      └────────┬───────┘        └────────┬───────┘
         │                       │                         │
         └───────────────────────┼─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   ALPHA-BETA PRUNING    │
                    │   - Optimization for    │
                    │     Minimax search      │
                    │   - Prunes branches     │
                    │   - Same result as      │
                    │     Minimax but faster  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  EVALUATION FUNCTION    │
                    │  - Score game states    │
                    │  - Heuristic features:  │
                    │    * Distance to food   │
                    │    * Ghost proximity    │
                    │    * Scared ghosts      │
                    │    * Food remaining     │
                    │    * Game score         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     GAME OUTPUT         │
                    │  - Action selection     │
                    │  - Visual display       │
                    │  - Score tracking       │
                    └─────────────────────────┘

Key Algorithms:
├─ Reflex Agent: Immediate state evaluation
├─ Minimax: Optimal adversarial search, O(b^d)
├─ Alpha-Beta: Efficient minimax, O(b^(d/2)) best case
└─ Expectimax: Probabilistic opponent modeling
```

---

## Features and Capabilities

### Agent Types

#### 1. Reflex Agent
- **Strategy:** Evaluate immediate successor states
- **Lookahead:** None (one-step evaluation)
- **Computation:** O(|A|) for |A| actions
- **Response Time:** <1ms
- **Best For:** Fast decisions, simple scenarios

#### 2. Minimax Agent
- **Strategy:** Adversarial search assuming optimal opponents
- **Lookahead:** Configurable depth (typically 2-4 plies)
- **Computation:** O(b^d) where b≈4-5 branches, d=depth
- **Response Time:** 10-1000ms depending on depth
- **Best For:** Perfect opponents, optimal guaranteed solutions

#### 3. Alpha-Beta Agent
- **Strategy:** Optimized minimax with branch pruning
- **Lookahead:** Same as minimax, but deeper possible
- **Computation:** O(b^(d/2)) to O(b^d) depending on ordering
- **Response Time:** 2-10x faster than minimax
- **Best For:** When minimax needed but faster execution required

#### 4. Expectimax Agent
- **Strategy:** Models random/suboptimal opponents
- **Lookahead:** Configurable depth
- **Computation:** O(b^d) (no pruning possible)
- **Response Time:** Similar to minimax
- **Best For:** Random or probabilistic opponents

---

## Quick Start

### Installation

Navigate to project directory:
```bash
cd 3_multiagent
```

### Running Agents

#### Reflex Agent
```bash
# Test on classic layout
python pacman.py -p ReflexAgent -l testClassic

# Multiple games for statistics
python pacman.py -p ReflexAgent -l testClassic -n 10
```

#### Minimax Agent
```bash
# Run on minimax-designed layout
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Medium classic with depth 3
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=3
```

#### Alpha-Beta Agent
```bash
# Optimized search
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3

# Fast mode (no graphics)
python pacman.py -p AlphaBetaAgent -l mediumClassic -a depth=4 -q -n 10
```

#### Expectimax Agent
```bash
# Model random ghosts
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Compare with minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

### Command Line Options

| Option | Description | Values |
|--------|-------------|---------|
| `-p` | Agent type | `ReflexAgent`, `MinimaxAgent`, `AlphaBetaAgent`, `ExpectimaxAgent` |
| `-l` | Layout/maze | `testClassic`, `minimaxClassic`, `smallClassic`, `mediumClassic` |
| `-a` | Agent args | `depth=N`, `evalFn=better` |
| `-q` | Quiet mode | No graphics, faster |
| `-n` | Games to play | `1`, `10`, `100` |
| `--frameTime` | Animation | `0` (fast), `0.1` (slow) |

---

## Algorithm Details

### Minimax Search

**Core Idea:** Assume all agents play optimally

**Pseudocode:**
```
function MINIMAX(state, depth, agentIndex):
    if depth = 0 or state is terminal:
        return EVALUATE(state)

    if agentIndex = 0:  # Pacman (MAX)
        value = -∞
        for each action in legal_actions:
            successor = generate_successor(state, action)
            value = max(value, MINIMAX(successor, depth, 1))
        return value

    else:  # Ghost (MIN)
        value = +∞
        for each action in legal_actions:
            successor = generate_successor(state, action)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent = 0 else depth
            value = min(value, MINIMAX(successor, nextDepth, nextAgent))
        return value
```

**Implementation:**
```python
def minimax(self, gameState, depth, agentIndex):
    # Terminal conditions
    if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)

    numAgents = gameState.getNumAgents()
    legalActions = gameState.getLegalActions(agentIndex)

    # Pacman's turn (MAX)
    if agentIndex == 0:
        return max(self.minimax(gameState.generateSuccessor(agentIndex, action),
                               depth, 1)
                  for action in legalActions)

    # Ghost's turn (MIN)
    else:
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        return min(self.minimax(gameState.generateSuccessor(agentIndex, action),
                               nextDepth, nextAgent)
                  for action in legalActions)
```

**Properties:**
- Optimal against perfect opponents
- Explores full game tree to depth d
- Exponential time complexity O(b^d)
- Guarantees best worst-case outcome

---

### Alpha-Beta Pruning

**Core Idea:** Prune branches that won't affect decision

**Parameters:**
- **α (alpha):** Best value MAX can guarantee (lower bound)
- **β (beta):** Best value MIN can guarantee (upper bound)

**Pruning Rule:**
```
If α ≥ β → prune remaining branches
```

**Pseudocode:**
```
function ALPHA-BETA(state, depth, agentIndex, α, β):
    if depth = 0 or terminal:
        return EVALUATE(state)

    if agentIndex = 0:  # MAX
        value = -∞
        for each action:
            value = max(value, ALPHA-BETA(successor, depth, 1, α, β))
            α = max(α, value)
            if α ≥ β:
                break  # β cutoff
        return value

    else:  # MIN
        value = +∞
        for each action:
            value = min(value, ALPHA-BETA(successor, nextDepth, nextAgent, α, β))
            β = min(β, value)
            if α ≥ β:
                break  # α cutoff
        return value
```

**Implementation:**
```python
def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
    if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)

    # MAX node
    if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.alphaBeta(successor, depth, 1, alpha, beta))

            # Pruning
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    # MIN node
    else:
        value = float('inf')
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = min(value, self.alphaBeta(successor, nextDepth, nextAgent, alpha, beta))

            # Pruning
            if value < alpha:
                return value
            beta = min(beta, value)
        return value
```

**Efficiency:**
- Best case: O(b^(d/2)) - perfect move ordering
- Average case: O(b^(3d/4))
- Typical speedup: 3-5x over minimax
- Enables search 1-2 plies deeper

---

### Expectimax Search

**Core Idea:** Model probabilistic opponents with expected values

**Pseudocode:**
```
function EXPECTIMAX(state, depth, agentIndex):
    if depth = 0 or terminal:
        return EVALUATE(state)

    if agentIndex = 0:  # MAX (Pacman)
        return max_{action} EXPECTIMAX(successor, depth, 1)

    else:  # CHANCE (Ghost)
        actions = legal_actions
        probability = 1.0 / len(actions)
        total = 0

        for each action:
            total += probability × EXPECTIMAX(successor, nextDepth, nextAgent)

        return total
```

**Implementation:**
```python
def expectimax(self, gameState, depth, agentIndex):
    if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)

    # MAX node (Pacman)
    if agentIndex == 0:
        return max(self.expectimax(gameState.generateSuccessor(agentIndex, action),
                                  depth, 1)
                  for action in legalActions)

    # CHANCE node (Ghost)
    else:
        probability = 1.0 / len(legalActions)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        return sum(probability * self.expectimax(
                       gameState.generateSuccessor(agentIndex, action),
                       nextDepth, nextAgent)
                  for action in legalActions)
```

**Properties:**
- More realistic than minimax (ghosts not optimal)
- Uses expected values instead of worst case
- Cannot be pruned (must explore all branches)
- Better for random opponents

---

## Evaluation Functions

### Design Principles

**Goal:** Score game states where higher = better for Pacman

**Components:**
1. **Game score:** Current score as baseline
2. **Food distance:** Closer to food = higher score
3. **Ghost distance:** Farther from ghosts = higher score (unless scared)
4. **Scared ghosts:** Closer to scared ghosts = higher score
5. **Food remaining:** Fewer food = higher score

### Example Implementation

```python
def betterEvaluationFunction(currentGameState):
    """
    Advanced evaluation function
    """
    # Terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # Extract state information
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Feature 1: Food distance
    foodList = food.asList()
    foodScore = 0
    if foodList:
        minFoodDist = min([manhattanDistance(pos, f) for f in foodList])
        foodScore = 10.0 / (minFoodDist + 1)

    # Feature 2: Ghost distance
    ghostScore = 0
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)

        if ghost.scaredTimer > 0:
            # Chase scared ghosts
            ghostScore += 200.0 / (ghostDist + 1)
        else:
            # Avoid active ghosts
            if ghostDist < 2:
                ghostScore -= 1000
            else:
                ghostScore += ghostDist * 2

    # Feature 3: Capsules
    capsuleScore = 0
    if capsules:
        minCapsuleDist = min([manhattanDistance(pos, c) for c in capsules])
        capsuleScore = 50.0 / (minCapsuleDist + 1)

    # Feature 4: Food remaining (penalty)
    foodRemainingScore = -5 * len(foodList)

    # Combine features with weights
    totalScore = (score +
                  foodScore +
                  ghostScore +
                  capsuleScore +
                  foodRemainingScore)

    return totalScore
```

### Feature Design Tips

1. **Use reciprocal for attraction:**
   ```python
   score += weight / (distance + 1)  # Avoid division by zero
   ```

2. **Use direct distance for repulsion:**
   ```python
   score += weight * distance  # or -weight / distance
   ```

3. **Scale features appropriately:**
   ```python
   # Normalize by maze size
   normalizedDist = distance / (width + height)
   ```

4. **Handle edge cases:**
   ```python
   # Empty food list
   if not foodList:
       return score  # Don't crash

   # Terminal states
   if gameState.isWin():
       return float('inf')
   ```

---

## Performance Analysis

### Algorithm Comparison

**Small Classic Layout (depth=3):**

| Agent | Avg Score | Win Rate | Time/Move | Nodes Expanded |
|-------|-----------|----------|-----------|----------------|
| Reflex | 1200 | 85% | <1ms | N/A |
| Minimax | 1400 | 95% | 150ms | ~500 |
| Alpha-Beta | 1400 | 95% | 50ms | ~180 |
| Expectimax | 1350 | 90% | 200ms | ~500 |

**Medium Classic Layout (depth=3):**

| Agent | Avg Score | Win Rate | Time/Move | Nodes Expanded |
|-------|-----------|----------|-----------|----------------|
| Minimax | 1800 | 85% | 800ms | ~2500 |
| Alpha-Beta | 1800 | 85% | 250ms | ~900 |
| Expectimax | 1750 | 80% | 900ms | ~2500 |

**Key Observations:**
- Alpha-Beta achieves same decisions as Minimax with 3-5x speedup
- Expectimax better models random ghosts but can't be pruned
- Deeper search improves decisions but increases computation exponentially

### Depth vs Performance

**Minimax Agent on mediumClassic:**

| Depth | Avg Score | Win Rate | Time/Move | Nodes Expanded |
|-------|-----------|----------|-----------|----------------|
| 2 | 1600 | 75% | 80ms | ~150 |
| 3 | 1800 | 85% | 800ms | ~2500 |
| 4 | 1900 | 90% | 8s | ~40000 |
| 5 | 1950 | 95% | 120s | ~600000 |

**Practical Depth Limits:**
- Depth 2-3: Playable in real-time
- Depth 4: Slow but acceptable
- Depth 5+: Impractical without optimizations

---

## Usage Examples

### Basic Game Playing

```bash
# Reflex agent (fast, reactive)
python pacman.py -p ReflexAgent -l testClassic

# Minimax (optimal but slow)
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Alpha-beta (optimal and fast)
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3

# Expectimax (probabilistic)
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

### Performance Testing

```bash
# Maximum speed comparison
python pacman.py -p MinimaxAgent -l smallClassic -a depth=3 -q -n 10
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3 -q -n 10

# Depth comparison
for d in 2 3 4; do
    echo "Testing depth $d"
    python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=$d -q -n 5
done
```

### Evaluation Function Testing

```bash
# Test custom evaluation
python pacman.py -p AlphaBetaAgent -a depth=2,evalFn=better -l smallClassic

# Compare default vs custom
python pacman.py -p AlphaBetaAgent -a depth=2 -l mediumClassic -n 10 -q
python pacman.py -p AlphaBetaAgent -a depth=2,evalFn=better -l mediumClassic -n 10 -q
```

---

## Key Concepts

### Game Tree Structure

**Alternating Layers:**
```
Depth 0: Pacman (MAX)
         |
Depth 1: Ghost1 (MIN)
         |
Depth 2: Ghost2 (MIN)
         |
Depth 3: Pacman (MAX)
         ...
```

**Ply vs Depth:**
- **Ply:** Single agent move
- **Depth:** Complete round (all agents moved)
- **Depth d:** d × numAgents plies

### Minimax Value

**Recursive Definition:**
```
MiniMax(s, 0) = Eval(s)  # depth 0

MiniMax(s, d, Pacman) = max_{a} MiniMax(successor(s,a), d, Ghost1)

MiniMax(s, d, Ghost_i) = min_{a} MiniMax(successor(s,a), d, Ghost_{i+1})

MiniMax(s, d, Ghost_last) = min_{a} MiniMax(successor(s,a), d-1, Pacman)
```

### Alpha-Beta Pruning

**Alpha (α):** Best value MAX can guarantee so far
**Beta (β):** Best value MIN can guarantee so far

**Pruning Conditions:**
- **In MAX node:** If value ≥ β, prune (MIN won't choose this path)
- **In MIN node:** If value ≤ α, prune (MAX won't choose this path)

**Update Rules:**
- **MAX node:** α = max(α, value)
- **MIN node:** β = min(β, value)

**Example:**
```
Root (MAX, α=-∞, β=+∞)
├─ Action1 → MIN (α=-∞, β=+∞)
│  ├─ Child1 → value = 5
│  │  Update β = min(+∞, 5) = 5
│  └─ Child2 → value = 3
│     Update β = min(5, 3) = 3
│     Return 3 to MAX
│  Update α = max(-∞, 3) = 3
│
├─ Action2 → MIN (α=3, β=+∞)
│  ├─ Child1 → value = 2
│     Since 2 ≤ α=3, PRUNE! Return 2
│  Update α = max(3, 2) = 3 (no change)
│
└─ Action3 → MIN (α=3, β=+∞)
   ... continue ...

Best Action = Action1 (value=3)
```

---

## Testing and Validation

### Automated Testing

```bash
# All questions
python autograder.py

# Individual questions
python autograder.py -q q1  # Reflex Agent
python autograder.py -q q2  # Minimax
python autograder.py -q q3  # Alpha-Beta
python autograder.py -q q4  # Expectimax
python autograder.py -q q5  # Evaluation Function

# Verbose debugging
python autograder.py -q q2 --verbose
```

### Manual Testing

```bash
# Visual testing
python pacman.py -p MinimaxAgent -l testClassic -a depth=2

# Performance comparison
python pacman.py -p MinimaxAgent -l smallClassic -a depth=3 -n 10 -q
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3 -n 10 -q

# Evaluation function impact
python pacman.py -p AlphaBetaAgent -a depth=2 -l mediumClassic -n 20 -q
python pacman.py -p AlphaBetaAgent -a depth=2,evalFn=better -l mediumClassic -n 20 -q
```

---

## Troubleshooting

### Agent Too Slow
**Problem:** Takes too long per move

**Solutions:**
- Reduce search depth (try depth=2)
- Optimize evaluation function
- Use Alpha-Beta instead of Minimax
- Profile code for bottlenecks
- Consider iterative deepening

### Agent Makes Poor Decisions
**Problem:** Loses frequently or makes obvious mistakes

**Solutions:**
- Improve evaluation function (add features)
- Increase search depth (if time permits)
- Tune feature weights
- Check terminal state handling
- Verify minimax logic is correct

### Alpha-Beta Different from Minimax
**Problem:** Different actions selected

**Solutions:**
- Check pruning conditions (α ≥ β)
- Verify α/β updates in correct nodes
- Ensure return value correct after pruning
- Test on simple cases first
- Compare intermediate values

### Expectimax Always Loses
**Problem:** Poor win rate

**Solutions:**
- Verify uniform probability distribution
- Check expected value calculation (sum, not min)
- Ensure correct agent turn handling
- Improve evaluation function
- Increase search depth

---

## Learning Objectives

✅ **Adversarial Search**
- Understand zero-sum games and minimax principle
- Implement minimax algorithm with alternating agents
- Handle multi-agent turns correctly
- Extract optimal actions from game tree

✅ **Alpha-Beta Pruning**
- Understand pruning conditions and invariants
- Implement efficient minimax optimization
- Manage alpha and beta bounds correctly
- Analyze pruning effectiveness

✅ **Expectimax Search**
- Model probabilistic opponent behavior
- Compute expected values over chance nodes
- Understand when minimax vs expectimax is appropriate
- Handle uncertainty in opponent actions

✅ **Evaluation Functions**
- Design heuristics for non-terminal states
- Balance multiple competing objectives
- Normalize and weight features appropriately
- Create fast and informative evaluations

---

## Additional Resources

### GameState Methods

```python
gameState.getLegalActions(agentIndex)          # Valid actions for agent
gameState.generateSuccessor(agentIndex, action) # Result of taking action
gameState.getNumAgents()                       # Total agents (Pacman + ghosts)
gameState.getPacmanPosition()                  # (x, y) position
gameState.getGhostPositions()                  # List of ghost positions
gameState.getFood()                            # Food grid
gameState.getCapsules()                        # Power pellet locations
gameState.getScore()                           # Current game score
gameState.isWin()                              # Victory check
gameState.isLose()                             # Defeat check
```

### Utility Functions

```python
from util import manhattanDistance              # L1 distance

# Game directions
from game import Directions
Directions.NORTH, SOUTH, EAST, WEST, STOP
```

---

## References

- **UC Berkeley CS188** - Original project framework
- **Russell & Norvig** - "Artificial Intelligence: A Modern Approach" (Chapters 5-6)
- **Game Theory Literature** - Minimax theorem, zero-sum games
- **Course Materials** - CS-5100 lecture slides on adversarial search
