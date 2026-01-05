# Multi-Agent Pacman Project

## Overview

This project implements **intelligent multi-agent systems** for the classic Pacman game. The system develops various AI agents that make decisions in adversarial environments with multiple agents (Pacman and ghosts). The project explores game-playing strategies through reflex agents, minimax algorithms, alpha-beta pruning, and expectimax search.

The implementation focuses on **game tree search algorithms** for adversarial scenarios where Pacman must navigate mazes while avoiding or eating ghosts. The agents demonstrate sophisticated decision-making capabilities using evaluation functions and tree search techniques.

## Multi-Agent System Architecture

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
├─ Minimax: Optimal adversarial search
├─ Alpha-Beta Pruning: Efficient minimax
└─ Expectimax: Probabilistic opponent modeling
```

## Features

### Core Capabilities

1. **Reflex Agent** - Reactive agent using state evaluation functions
2. **Minimax Agent** - Adversarial search for optimal play against perfect opponents
3. **Alpha-Beta Agent** - Optimized minimax with branch pruning
4. **Expectimax Agent** - Handles probabilistic opponent behavior
5. **Evaluation Functions** - Sophisticated state scoring heuristics

### Agent Types

#### 1. Reflex Agent
- Makes decisions based on immediate game state evaluation
- No lookahead or tree search
- Fast response time
- Good for simple scenarios

#### 2. Minimax Agent
- Assumes optimal play by all agents
- Explores full game tree to specified depth
- Alternates between maximizing (Pacman) and minimizing (ghosts) layers
- Guarantees optimal strategy against perfect opponents

#### 3. Alpha-Beta Agent
- Optimized version of Minimax
- Prunes branches that won't affect final decision
- Same results as Minimax but significantly faster
- Enables deeper search within time constraints

#### 4. Expectimax Agent
- Models suboptimal/random opponent behavior
- Uses expected values instead of min nodes
- Better for realistic ghost behaviors
- Handles uncertainty in opponent actions

### Evaluation Features

The system uses multi-dimensional state evaluation considering:

1. **Food Distance** - Proximity to nearest food pellet
2. **Ghost Distance** - Safety from active ghosts
3. **Scared Ghosts** - Opportunity to eat frightened ghosts
4. **Food Count** - Remaining food pellets
5. **Capsules** - Power pellet availability
6. **Score** - Current game score

## Quick Start

### Requirements
- **Python 3.6+** - For running the game and agents
- All dependencies included in project files
- No external packages required

### Running the Agents

#### Reflex Agent
```bash
python pacman.py -p ReflexAgent -l testClassic
```

#### Minimax Agent
```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

#### Alpha-Beta Agent
```bash
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3
```

#### Expectimax Agent
```bash
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

### Command Line Options

**Parameters:**
- `-p` - Agent type (ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent)
- `-l` - Layout/maze (testClassic, minimaxClassic, smallClassic, mediumClassic, etc.)
- `-a depth=N` - Search depth for tree-based agents (default: 2)
- `-q` - Quiet mode (no graphics)
- `-n` - Number of games to play
- `--frameTime=0` - Run at maximum speed

**Examples:**
```bash
# Run Minimax agent with depth 4 on medium classic maze
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=4

# Run Alpha-Beta agent quickly without graphics
python pacman.py -p AlphaBetaAgent -q -n 10

# Test evaluation function
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=2,evalFn=better
```

## Usage

### 1. Testing Reflex Agent

The reflex agent evaluates immediate successor states:

```bash
python pacman.py -p ReflexAgent -l testClassic
```

**Key Behaviors:**
- Moves toward nearest food
- Avoids active ghosts
- Chases scared ghosts
- Considers immediate consequences only

### 2. Running Minimax Agent

Minimax agent performs adversarial search:

```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

**Parameters:**
- `depth` - Number of plies to search (higher = better but slower)

**Typical Performance:**
- Depth 2: Fast, basic strategy
- Depth 3: Good balance
- Depth 4: Strong play, slower
- Depth 5+: Very slow, diminishing returns

### 3. Alpha-Beta Pruning

Optimized adversarial search:

```bash
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3
```

**Benefits:**
- Same decisions as Minimax
- Significantly faster (often 2-5x speedup)
- Enables deeper search depths
- More nodes pruned = better performance

### 4. Expectimax Agent

Models probabilistic opponents:

```bash
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

**Use Cases:**
- Random ghost movement
- Suboptimal opponent play
- Uncertain environments
- More realistic game modeling

### 5. Evaluation Function Testing

Test custom evaluation functions:

```bash
python pacman.py -p AlphaBetaAgent -a depth=2,evalFn=better -l smallClassic
```

## How It Works

### Game Tree Search

#### Minimax Algorithm

```
Algorithm: Minimax(state, depth, agentIndex)
1. If depth = 0 or state is terminal:
       return evaluationFunction(state)
2. If agentIndex = 0 (Pacman - MAX):
       value = -infinity
       for each legal action:
           successor = generateSuccessor(state, action)
           value = max(value, Minimax(successor, depth, 1))
       return value
3. Else (Ghost - MIN):
       value = +infinity
       for each legal action:
           successor = generateSuccessor(state, action)
           nextAgent = (agentIndex + 1) % numAgents
           nextDepth = depth - 1 if nextAgent = 0 else depth
           value = min(value, Minimax(successor, nextDepth, nextAgent))
       return value
```

**Time Complexity:** O(b^d) where b = branching factor, d = depth
**Space Complexity:** O(bd) for recursion stack

#### Alpha-Beta Pruning

```
Algorithm: AlphaBeta(state, depth, agentIndex, alpha, beta)
1. If depth = 0 or state is terminal:
       return evaluationFunction(state)
2. If agentIndex = 0 (Pacman - MAX):
       value = -infinity
       for each legal action:
           successor = generateSuccessor(state, action)
           value = max(value, AlphaBeta(successor, depth, 1, alpha, beta))
           alpha = max(alpha, value)
           if alpha >= beta:
               break  # Beta cutoff
       return value
3. Else (Ghost - MIN):
       value = +infinity
       for each legal action:
           successor = generateSuccessor(state, action)
           nextAgent = (agentIndex + 1) % numAgents
           nextDepth = depth - 1 if nextAgent = 0 else depth
           value = min(value, AlphaBeta(successor, nextDepth, nextAgent, alpha, beta))
           beta = min(beta, value)
           if alpha >= beta:
               break  # Alpha cutoff
       return value
```

**Pruning Rules:**
- **Alpha**: Best value MAX can guarantee
- **Beta**: Best value MIN can guarantee
- **Cutoff**: When alpha ≥ beta, remaining branches can be pruned

**Efficiency:** Prunes up to 50-90% of nodes in best-case ordering

#### Expectimax Algorithm

```
Algorithm: Expectimax(state, depth, agentIndex)
1. If depth = 0 or state is terminal:
       return evaluationFunction(state)
2. If agentIndex = 0 (Pacman - MAX):
       value = -infinity
       for each legal action:
           successor = generateSuccessor(state, action)
           value = max(value, Expectimax(successor, depth, 1))
       return value
3. Else (Ghost - CHANCE):
       totalValue = 0
       actions = getLegalActions(state, agentIndex)
       probability = 1.0 / len(actions)
       for each legal action:
           successor = generateSuccessor(state, action)
           nextAgent = (agentIndex + 1) % numAgents
           nextDepth = depth - 1 if nextAgent = 0 else depth
           totalValue += probability × Expectimax(successor, nextDepth, nextAgent)
       return totalValue
```

**Key Difference:** Uses expected value (average) instead of minimum for ghost moves

### Evaluation Function Design

A good evaluation function considers:

1. **Linear Combination of Features:**
```
score = w1×food_score + w2×ghost_score + w3×capsule_score + w4×game_score
```

2. **Food Score:**
- Distance to nearest food (negative weight)
- Total food remaining (negative weight)
- Encourages eating food

3. **Ghost Score:**
- Distance to nearest active ghost (positive weight if far)
- Distance to scared ghosts (negative weight - pursue them)
- Immediate danger detection

4. **Game State Score:**
- Current game score (positive weight)
- Win state (very high score)
- Lose state (very low score)

5. **Strategic Considerations:**
- Prefer actions leading to food collection
- Maintain safe distance from ghosts
- Exploit scared ghost opportunities
- Balance exploration vs exploitation

## Algorithm Workflow Example

### Minimax Decision Tree Example

**Scenario:** Pacman at position, 2 ghosts nearby, depth=2

```
                        State0 (Pacman)
                    Score = Minimax(depth=2)
                              |
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    Move North          Move South            Move East
        │                     │                     │
    State1-N             State1-S             State1-E
   (Ghost1)             (Ghost1)             (Ghost1)
   Score=MIN            Score=MIN            Score=MIN
        │                     │                     │
   ┌────┼────┐          ┌────┼────┐          ┌────┼────┐
   │    │    │          │    │    │          │    │    │
  G1N  G1S  G1E       G1N  G1S  G1E       G1N  G1S  G1E
   │    │    │          │    │    │          │    │    │
State2  │    │       State2  │    │       State2  │    │
(G2)    │    │       (G2)    │    │       (G2)    │    │
MIN     │    │       MIN     │    │       MIN     │    │
        │    │                │    │               │    │
    [Evaluate]          [Evaluate]            [Evaluate]

Evaluation at depth 2:
- Each leaf node: evaluate(state)
- Ghost2 layer: min(children)
- Ghost1 layer: min(children)
- Pacman root: max(children) → Best action
```

### Alpha-Beta Pruning Example

```
                    Root (Pacman, α=-∞, β=+∞)
                              |
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    Move N (α=-∞)       Move S (α=5)          Move E (PRUNED!)
        │                     │
    MIN (β=+∞)          MIN (β=+∞)            [Not explored]
        │                     │
   ┌────┼────┐          ┌────┼────┐
Eval=5  Eval=3  (stop) Eval=7  Eval=4

MIN(5,3)=3              MIN(7,4)=4
Update α=max(-∞,3)=3    Update α=max(3,4)=4

Move E: β would be ≤4, but α=4
Since α ≥ β would be true for any value ≤4, prune!
```

**Pruning saved:** Entire subtree under Move E (multiple nodes)

## Project Structure

```
multiagent/
├── multiAgents.py           # Main agent implementations
├── pacman.py                # Main game engine
├── game.py                  # Game state and rules
├── ghostAgents.py           # Ghost behaviors
├── graphicsDisplay.py       # Visual display
├── graphicsUtils.py         # Display utilities
├── textDisplay.py           # Text-based display
├── keyboardAgents.py        # Human control
├── layout.py                # Maze layouts
├── layouts/                 # Maze layout files
│   ├── testClassic.lay
│   ├── minimaxClassic.lay
│   ├── smallClassic.lay
│   └── mediumClassic.lay
├── util.py                  # Data structures
├── autograder.py            # Automated testing
├── test_cases/              # Test cases
└── README.md                # This file
```

### Key Files

**Agent Implementations:**
- `multiAgents.py` - ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent

**Game Engine:**
- `pacman.py` - Main game loop and command-line interface
- `game.py` - Core game mechanics and state management
- `ghostAgents.py` - Ghost AI behaviors

**Testing:**
- `autograder.py` - Automated grading system
- `test_cases/` - Test scenarios for each agent

## Performance Considerations

### Reflex Agent
- **Time Complexity:** O(n) where n = number of legal actions
- **Space Complexity:** O(1)
- **Performance:** Instant response (<1ms)

### Minimax Agent
- **Time Complexity:** O(b^d) where b = branching factor (~4-5), d = depth
- **Space Complexity:** O(bd)
- **Performance:**
  - Depth 2: ~10-50ms per move
  - Depth 3: ~50-200ms per move
  - Depth 4: ~200-1000ms per move

### Alpha-Beta Agent
- **Best Case:** O(b^(d/2)) - perfect move ordering
- **Average Case:** O(b^(3d/4))
- **Worst Case:** O(b^d) - same as Minimax
- **Performance:** 2-10x faster than Minimax (typical 3-5x)

### Expectimax Agent
- **Time Complexity:** O(b^d) - no pruning possible
- **Performance:** Similar to Minimax, slightly slower due to probability calculations

### Optimization Tips

1. **Better Evaluation Functions:**
   - Faster computation
   - More accurate state assessment
   - Reduces need for deep search

2. **Move Ordering:**
   - Try promising moves first
   - Increases Alpha-Beta pruning efficiency
   - Can use simple heuristics

3. **Transposition Tables:**
   - Cache evaluated positions
   - Avoid redundant computation
   - Significant speedup for repetitive states

4. **Iterative Deepening:**
   - Start with shallow search
   - Progressively deepen
   - Ensures response within time limit

## Testing

### Run Autograder

```bash
# Test all questions
python autograder.py

# Test specific question
python autograder.py -q q1
python autograder.py -q q2
python autograder.py -q q3
```

### Test Individual Agents

```bash
# Test Reflex Agent
python autograder.py -q q1

# Test Minimax Agent
python autograder.py -q q2

# Test Alpha-Beta Agent
python autograder.py -q q3

# Test Expectimax Agent
python autograder.py -q q4

# Test Evaluation Function
python autograder.py -q q5
```

## Troubleshooting

### Agent Too Slow
**Problem:** Agent takes too long to make decisions

**Solutions:**
1. Reduce search depth
2. Optimize evaluation function
3. Use Alpha-Beta instead of Minimax
4. Profile code for bottlenecks

### Agent Loses Frequently
**Problem:** Agent makes poor decisions

**Solutions:**
1. Improve evaluation function
2. Increase search depth (if time permits)
3. Add more features (ghost distance, food clustering)
4. Tune feature weights

### Minimax vs Alpha-Beta Different Results
**Problem:** Alpha-Beta gives different answers than Minimax

**Solutions:**
1. Check pruning logic
2. Verify alpha/beta initialization
3. Ensure correct return values
4. Test on simple cases first

### Expectimax Always Loses
**Problem:** Expectimax performs poorly

**Solutions:**
1. Verify probability distribution (uniform)
2. Check expected value calculation
3. Ensure correct agent turn handling
4. Test evaluation function separately

## License

This project is part of academic coursework for CS-5100 at Northeastern University.

**Attribution:** The Pacman AI projects were developed at UC Berkeley. The core projects and autograders were primarily created by John DeNero and Dan Klein.

### 📚 Usage as Reference

This repository is intended as a **learning resource and reference guide**. If you're working on a similar project:

- Use it to understand algorithm implementations and approaches
- Reference it when debugging your own code or stuck on concepts
- Learn from the structure and design patterns

Please respect academic integrity policies at your institution. This code should guide your learning, not replace it. Write your own implementations and cite references appropriately.

## Acknowledgments

- **UC Berkeley CS188** for the Pacman AI framework
- **Professor John DeNero and Dan Klein** for the original project design
- **CS-5100 course staff** for guidance and project specifications
