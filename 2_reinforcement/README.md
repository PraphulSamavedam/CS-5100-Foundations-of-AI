# Reinforcement Learning Project

## Overview

This project implements **Reinforcement Learning (RL) agents** that learn optimal policies through interaction with environments. The system explores value-based learning methods including Value Iteration, Q-Learning, and Approximate Q-Learning, demonstrating how agents can learn from experience without explicit instruction.

The implementation covers multiple learning scenarios from simple gridworlds to complex Pacman environments. Agents learn through trial and error, updating their knowledge based on rewards and penalties, ultimately discovering optimal strategies for navigation, resource collection, and game playing.

## Reinforcement Learning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  REINFORCEMENT LEARNING SYSTEM                              │
│                       System Workflow Diagram                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │      ENVIRONMENT                 │
                    │  - Gridworld                     │
                    │  - Crawler Robot                 │
                    │  - Pacman Game                   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   STATE & ACTIONS            │
                    │  - Current State (s)         │
                    │  - Available Actions (a)     │
                    │  - Transition Model T(s,a,s')│
                    │  - Reward Function R(s,a,s') │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼────────────────────────────┐
        │                          │                            │
        ▼                          ▼                            ▼
┌────────────────┐      ┌────────────────┐        ┌────────────────┐
│ VALUE          │      │ Q-LEARNING     │        │ APPROXIMATE    │
│ ITERATION      │      │                │        │ Q-LEARNING     │
│                │      │ - Model-free   │        │                │
│ - Offline      │      │ - Temporal     │        │ - Function     │
│   planning     │      │   difference   │        │   approximation│
│ - Full model   │      │ - Experience   │        │ - Feature-based│
│ - Bellman eqn  │      │   replay       │        │ - Generalization│
└────────┬───────┘      └────────┬───────┘        └────────┬───────┘
         │                       │                         │
         └───────────────────────┼─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   POLICY EXTRACTION     │
                    │   π(s) = argmax_a Q(s,a)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LEARNED POLICY        │
                    │   - Optimal actions     │
                    │   - Value estimates     │
                    │   - State preferences   │
                    └─────────────────────────┘

Key Algorithms:
├─ Value Iteration: V(s) = max_a Σ T(s,a,s')[R(s,a,s') + γV(s')]
├─ Q-Learning: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
└─ Approximate Q-Learning: Q(s,a) = Σ w_i × f_i(s,a)
```

## Features

### Core Capabilities

1. **Value Iteration Agent** - Offline planning using dynamic programming
2. **Q-Learning Agent** - Model-free temporal difference learning
3. **Approximate Q-Learning** - Function approximation with feature-based learning
4. **Epsilon-Greedy Exploration** - Balancing exploration vs exploitation
5. **Experience Replay** - Learning from past experiences

### Learning Algorithms

#### 1. Value Iteration
- **Type:** Model-based, offline planning
- **Use Case:** Known transition model and rewards
- **Method:** Dynamic programming on state values
- **Convergence:** Guaranteed with sufficient iterations

#### 2. Q-Learning
- **Type:** Model-free, online learning
- **Use Case:** Unknown environment dynamics
- **Method:** Temporal difference updates on Q-values
- **Convergence:** Guaranteed with proper learning schedule

#### 3. Approximate Q-Learning
- **Type:** Model-free with function approximation
- **Use Case:** Large or continuous state spaces
- **Method:** Linear combination of features
- **Advantage:** Generalizes across similar states

### Environments

#### 1. Gridworld
- **Description:** Simple grid navigation
- **States:** Grid cells (x, y)
- **Actions:** North, South, East, West
- **Rewards:** Goal states (+10), cliffs (-10), living cost (-0.1)
- **Purpose:** Visualizing learning algorithms

#### 2. Crawler Robot
- **Description:** 2-link robot arm locomotion
- **States:** Joint angles (arm, hand)
- **Actions:** Increment/decrement joint angles
- **Rewards:** Forward movement distance
- **Purpose:** Continuous control learning

#### 3. Pacman
- **Description:** Classic Pacman game
- **States:** Maze position, ghost locations, food
- **Actions:** North, South, East, West
- **Rewards:** Food collection (+10), ghost penalties (-500), win (+500)
- **Purpose:** Complex game learning

## Quick Start

### Requirements
- **Python 3.6+** - For running learning agents
- All dependencies included in project files
- No external packages required

### Running Learning Agents

#### Value Iteration
```bash
# Run on gridworld
python gridworld.py -a value -i 100 -k 10

# Parameters:
# -i: Number of iterations (default: 10)
# -k: Number of episodes to run
```

#### Q-Learning
```bash
# Train Q-learning agent on gridworld
python gridworld.py -a q -k 100

# With custom parameters
python gridworld.py -a q -k 100 -e 0.1 -l 0.5 -g 0.9
```

#### Approximate Q-Learning on Pacman
```bash
# Train Pacman agent
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60

# Parameters:
# -x: Number of training episodes
# -n: Total episodes (training + testing)
```

### Command Line Options

**General Parameters:**
- `-a` - Agent type (value, q, random)
- `-k` - Number of episodes
- `-g` - Discount factor (gamma, default: 0.9)
- `-l` - Learning rate (alpha, default: 0.5)
- `-e` - Epsilon for exploration (default: 0.1)
- `-q` - Quiet mode (no graphics)
- `-n` - Number of games

**Pacman-Specific:**
- `-x` - Number of training games
- `-p` - Agent type (PacmanQAgent, ApproximateQAgent)
- `--frameTime=0` - Run at maximum speed

## Usage

### 1. Value Iteration on Gridworld

Compute optimal values using dynamic programming:

```bash
python gridworld.py -a value -i 100
```

**Controls:**
- Arrow keys to manually control
- Press 'a' to auto-run policy
- Press 'q' to quit

**Visualization:**
- State values shown in each cell
- Optimal policy shown with arrows
- Updates happen in real-time

**Parameters:**
- `-i N` - Run N iterations of value iteration
- `-g G` - Set discount factor to G (0 < G < 1)

**Example:**
```bash
# Discount factor 0.9, 100 iterations
python gridworld.py -a value -i 100 -g 0.9
```

### 2. Q-Learning on Gridworld

Learn through experience:

```bash
python gridworld.py -a q -k 100
```

**Learning Process:**
- Agent starts with no knowledge
- Learns from trial and error
- Updates Q-values after each transition
- Gradually improves policy

**Parameters:**
- `-k N` - Run N episodes
- `-l L` - Learning rate (alpha)
- `-e E` - Exploration rate (epsilon)
- `-g G` - Discount factor (gamma)

**Example:**
```bash
# 200 episodes, learning rate 0.3, exploration 0.1
python gridworld.py -a q -k 200 -l 0.3 -e 0.1
```

### 3. Crawler Robot

Train robot to move forward:

```bash
python crawler.py
```

**Visualization:**
- Robot shown with two joints
- Q-values displayed for each action
- Cumulative reward shown
- Learning progress tracked

**Controls:**
- GUI controls for manual testing
- Watch agent learn optimal gait
- Speed control for visualization

### 4. Q-Learning on Pacman

```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

**Training Process:**
- First 2000 games: training (exploring)
- Last 10 games: testing (exploiting)
- Q-values saved after training

**Parameters:**
- `-x N` - N training episodes
- `-n M` - Total episodes (M > N)
- `-l LAYOUT` - Maze layout
- `-a epsilon=E` - Exploration rate

### 5. Approximate Q-Learning on Pacman

Learn with feature-based representation:

```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

**Feature Extractors:**
- **SimpleExtractor** - Basic features (ghost distance, food)
- **Custom extractors** - Define your own features

**Advantages:**
- Faster learning
- Better generalization
- Works on unseen states
- Scalable to large environments

**Example Training:**
```bash
# Train for 50 games, test for 10
python pacman.py -p ApproximateQAgent -x 50 -n 60
```

## How It Works

### 1. Value Iteration Algorithm

**Bellman Update:**
```
V_{k+1}(s) = max_a Σ_{s'} T(s,a,s') [R(s,a,s') + γ × V_k(s')]
```

**Algorithm:**
```
1. Initialize V_0(s) = 0 for all states s
2. For k = 1 to K:
       For each state s:
           V_{k+1}(s) = max_a Q_k(s,a)
           where Q_k(s,a) = Σ_{s'} T(s,a,s')[R(s,a,s') + γ V_k(s')]
3. Extract policy: π(s) = argmax_a Q(s,a)
```

**Time Complexity:** O(K × |S|² × |A|) where:
- K = number of iterations
- |S| = number of states
- |A| = number of actions

**Convergence:** Values converge to V* (optimal values)

### 2. Q-Learning Algorithm

**Temporal Difference Update:**
```
Q(s,a) ← Q(s,a) + α [R(s,a,s') + γ max_{a'} Q(s',a') - Q(s,a)]
```

**Components:**
- **α (alpha):** Learning rate (0 < α ≤ 1)
- **γ (gamma):** Discount factor (0 ≤ γ < 1)
- **ε (epsilon):** Exploration rate for ε-greedy policy

**Algorithm:**
```
1. Initialize Q(s,a) = 0 for all s,a
2. For each episode:
       s = initial_state
       While not terminal:
           # Exploration vs Exploitation
           if random() < ε:
               a = random_action()
           else:
               a = argmax_a Q(s,a)

           # Execute action
           s', r = environment.step(s, a)

           # TD Update
           sample = r + γ × max_{a'} Q(s',a')
           Q(s,a) ← (1-α)Q(s,a) + α × sample

           s = s'
```

**Key Properties:**
- **Model-free:** No need to know T(s,a,s') or R(s,a,s')
- **Online:** Learns while acting
- **Off-policy:** Can learn optimal policy while exploring
- **Convergence:** Guaranteed with decreasing α and sufficient exploration

### 3. Approximate Q-Learning

**Linear Function Approximation:**
```
Q(s,a) = w_1×f_1(s,a) + w_2×f_2(s,a) + ... + w_n×f_n(s,a)
       = Σ_i w_i × f_i(s,a)
```

**Weight Update:**
```
difference = R(s,a,s') + γ max_{a'} Q(s',a') - Q(s,a)
w_i ← w_i + α × difference × f_i(s,a)
```

**Algorithm:**
```
1. Initialize weights w_i = 0 for all features
2. For each episode:
       s = initial_state
       While not terminal:
           # Action selection
           a = ε-greedy(s)

           # Execute
           s', r = environment.step(s, a)

           # Compute features
           features = featureExtractor(s, a)

           # Compute difference
           prediction = Σ_i w_i × features[i]
           target = r + γ × max_{a'} Q(s',a')
           difference = target - prediction

           # Update weights
           for i, f_i in enumerate(features):
               w_i ← w_i + α × difference × f_i

           s = s'
```

**Advantages:**
- **Generalization:** Similar states → similar Q-values
- **Efficiency:** Learn from fewer samples
- **Scalability:** Handle large state spaces
- **Transfer:** Knowledge transfers across states

**Feature Design Principles:**
1. **Relevance:** Features should capture important state aspects
2. **Discriminative:** Different states → different feature values
3. **Efficient:** Fast to compute
4. **Normalized:** Similar scales for all features

### Example: Pacman Features

```python
def getFeatures(state, action):
    features = util.Counter()

    # Feature 1: Bias term
    features["bias"] = 1.0

    # Feature 2: Distance to nearest food
    successor = state.generateSuccessor(0, action)
    foodList = successor.getFood().asList()
    if foodList:
        minDistance = min([manhattanDistance(successor.getPacmanPosition(), food)
                          for food in foodList])
        features["closest-food"] = float(minDistance) / (walls.width * walls.height)

    # Feature 3: Number of ghosts 1 step away
    ghosts = [successor.getGhostPosition(i) for i in range(1, successor.getNumAgents())]
    features["#-of-ghosts-1-step-away"] = sum(
        manhattanDistance(successor.getPacmanPosition(), g) <= 1
        for g in ghosts
    )

    # Feature 4: Eating food
    features["eats-food"] = 1.0 if successor.getFood()[x][y] else 0.0

    return features
```

## Algorithm Workflow Example

### Q-Learning Episode

**Scenario:** Agent in 3x3 gridworld, goal at (2,2)

```
Episode 1:
Initial State: (0,0), Q-values all 0

Step 1:
State: (0,0)
Action: East (ε-greedy, ε=0.1)
Reward: -0.1 (living cost)
Next State: (1,0)

Update:
Q((0,0), East) ← 0 + 0.5 × [-0.1 + 0.9 × 0 - 0] = -0.05

Step 2:
State: (1,0)
Action: North
Reward: -0.1
Next State: (1,1)

Update:
Q((1,0), North) ← 0 + 0.5 × [-0.1 + 0 - 0] = -0.05

...continues until goal

Final Step:
State: (2,1)
Action: North
Reward: +10 (goal)
Next State: (2,2) [terminal]

Update:
Q((2,1), North) ← 0 + 0.5 × [10 + 0 - 0] = 5.0

Episode 2:
Now Q((2,1), North) = 5.0
This value will propagate backwards through more updates
```

**After Many Episodes:**
```
Q-values converge:
Q((0,0), East) ≈ 7.2
Q((0,0), North) ≈ 6.8
Q((1,0), East) ≈ 7.8
Q((1,0), North) ≈ 8.2
...

Optimal Policy:
(0,0) → East
(1,0) → North or East
(1,1) → East or North
(2,1) → North
```

## Performance Considerations

### Value Iteration
- **Computation:** O(|S|² × |A|) per iteration
- **Convergence:** Usually 10-100 iterations
- **Memory:** O(|S|) for storing values
- **Use When:** Model known, offline planning acceptable

### Q-Learning
- **Training Time:** Thousands of episodes often needed
- **Memory:** O(|S| × |A|) for Q-table
- **Sample Efficiency:** Requires many samples
- **Use When:** Model-free learning, online updates needed

### Approximate Q-Learning
- **Training Time:** Faster than tabular Q-learning
- **Memory:** O(n) for n feature weights
- **Generalization:** Excellent across similar states
- **Use When:** Large state spaces, function approximation beneficial

### Hyperparameter Tuning

#### Learning Rate (α)
- **High (0.5-1.0):** Fast learning, unstable
- **Medium (0.1-0.5):** Balanced
- **Low (0.01-0.1):** Slow learning, stable
- **Decay:** Start high, decrease over time

#### Discount Factor (γ)
- **High (0.95-0.99):** Long-term planning
- **Medium (0.8-0.95):** Balanced
- **Low (0.5-0.8):** Myopic, immediate rewards

#### Exploration Rate (ε)
- **High (0.3-0.5):** More exploration (early training)
- **Medium (0.1-0.3):** Balanced
- **Low (0.01-0.1):** More exploitation (late training)
- **Decay:** ε-greedy → ε = ε × decay_rate

## Project Structure

```
reinforcement/
├── valueIterationAgents.py     # Value iteration implementation
├── qlearningAgents.py          # Q-learning agents
│   ├── QLearningAgent          # Tabular Q-learning
│   └── ApproximateQAgent       # Approximate Q-learning
├── learningAgents.py           # Base learning agent classes
├── featureExtractors.py        # Feature extraction for approximation
├── analysis.py                 # Analysis questions
│
├── Environment simulators:
├── gridworld.py                # Gridworld environment
├── crawler.py                  # Crawler robot simulation
├── pacman.py                   # Pacman game
│
├── Supporting files:
├── game.py                     # Game state management
├── util.py                     # Data structures
├── layout.py                   # Maze layouts
├── layouts/                    # Maze files
│
├── Testing:
├── autograder.py               # Automated testing
├── test_cases/                 # Test scenarios
│
└── README.md                   # This file
```

## Testing

### Automated Testing

```bash
# Test all questions
python autograder.py

# Test specific questions
python autograder.py -q q1  # Value Iteration
python autograder.py -q q2  # Q-learning - Bridge
python autograder.py -q q3  # Q-learning - Gridworld
python autograder.py -q q4  # Q-learning - Pacman
python autograder.py -q q5  # Approximate Q-learning
```

### Manual Testing

```bash
# Value iteration with different parameters
python gridworld.py -a value -i 100 -g 0.9

# Q-learning convergence
python gridworld.py -a q -k 1000 -e 0.1

# Pacman training visualization
python pacman.py -p ApproximateQAgent -x 50 -n 60 --frameTime=0.1
```

## Troubleshooting

### Q-Learning Not Converging
**Problem:** Q-values don't stabilize

**Solutions:**
1. Increase training episodes
2. Decrease learning rate
3. Ensure sufficient exploration (ε)
4. Check for negative cycles

### Pacman Performs Poorly
**Problem:** Low win rate after training

**Solutions:**
1. Train for more episodes (try 2000+)
2. Adjust feature weights
3. Add more informative features
4. Check feature normalization

### Approximate Q-Learning Doesn't Generalize
**Problem:** Poor performance on unseen states

**Solutions:**
1. Design better features
2. Normalize feature values
3. Increase feature diversity
4. Ensure features capture relevant info

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
- **Richard Sutton and Andrew Barto** for "Reinforcement Learning: An Introduction"
- **CS-5100 course staff** for guidance and project specifications
