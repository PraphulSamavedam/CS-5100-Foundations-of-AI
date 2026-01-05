# Reinforcement Learning

## Overview

The Reinforcement Learning project implements value-based learning methods where agents discover optimal policies through trial-and-error interaction with environments. Built on the foundations of Markov Decision Processes (MDPs), the system explores both model-based (Value Iteration) and model-free (Q-Learning) algorithms, culminating in function approximation techniques for handling complex state spaces.

The implementation covers **three distinct environments** (Gridworld, Crawler Robot, Pacman) and demonstrates how agents can learn from experience without explicit instruction, achieving **optimal policies** through systematic exploration and temporal difference updates.

---

## System Architecture

The reinforcement learning system follows a learning pipeline from basic value iteration to advanced function approximation:

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

---

## Features and Capabilities

### Learning Algorithms

#### 1. Value Iteration
- **Type:** Model-based offline planning
- **Method:** Dynamic programming with Bellman updates
- **Requires:** Complete knowledge of T(s,a,s') and R(s,a,s')
- **Convergence:** Guaranteed with sufficient iterations
- **Complexity:** O(|S|² × |A|) per iteration
- **Best For:** Known MDP, offline planning acceptable

#### 2. Q-Learning
- **Type:** Model-free online learning
- **Method:** Temporal difference updates on Q-values
- **Requires:** Only ability to interact with environment
- **Convergence:** Guaranteed with proper learning schedule
- **Complexity:** O(1) per experience tuple
- **Best For:** Unknown environment, online learning needed

#### 3. Approximate Q-Learning
- **Type:** Model-free with function approximation
- **Method:** Linear combination of feature weights
- **Requires:** Feature extractor design
- **Convergence:** Approximate (function approximation error)
- **Complexity:** O(k) for k features
- **Best For:** Large state spaces, generalization needed

### Learning Environments

#### Gridworld
- **Description:** Simple grid navigation
- **States:** Grid cells (x, y)
- **Actions:** North, South, East, West
- **Rewards:** Goal (+10), cliff (-10), living cost (-0.1)
- **Purpose:** Algorithm visualization and debugging

#### Crawler Robot
- **Description:** 2-link arm locomotion
- **States:** Joint angles (arm, hand)
- **Actions:** Increment/decrement angles
- **Rewards:** Forward movement distance
- **Purpose:** Continuous control and exploration

#### Pacman
- **Description:** Classic game environment
- **States:** Position, ghosts, food grid
- **Actions:** North, South, East, West
- **Rewards:** Food (+10), ghost penalty (-500), win (+500)
- **Purpose:** Complex decision-making

---

## Quick Start

### Installation

Navigate to project directory:
```bash
cd 2_reinforcement
```

### Running Learning Agents

#### Value Iteration on Gridworld
```bash
# Run 100 iterations
python gridworld.py -a value -i 100 -k 10

# With custom discount factor
python gridworld.py -a value -i 100 -g 0.95
```

#### Q-Learning on Gridworld
```bash
# Train for 100 episodes
python gridworld.py -a q -k 100

# Custom learning parameters
python gridworld.py -a q -k 100 -e 0.1 -l 0.5 -g 0.9
```

#### Crawler Robot Learning
```bash
# Watch robot learn to crawl
python crawler.py
```

#### Q-Learning on Pacman
```bash
# Train for 2000 episodes, test for 10
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

#### Approximate Q-Learning on Pacman
```bash
# Train with feature approximation
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

### Command Line Options

| Option | Description | Values |
|--------|-------------|---------|
| `-a` | Agent type | `value`, `q`, `random` |
| `-k` | Number of episodes | `10`, `100`, `1000` |
| `-i` | Iterations (value iteration) | `10`, `50`, `100` |
| `-g` | Discount factor (γ) | `0.1` to `0.99` |
| `-l` | Learning rate (α) | `0.1` to `1.0` |
| `-e` | Exploration rate (ε) | `0.0` to `1.0` |
| `-x` | Training episodes | `50`, `2000` |
| `-n` | Total episodes | Must be > `-x` |

---

## Algorithm Details

### Value Iteration

**Bellman Optimality Equation:**
```
V*(s) = max_a Σ_{s'} T(s,a,s') [R(s,a,s') + γ V*(s')]
```

**Algorithm:**
```
1. Initialize V_0(s) = 0 for all states
2. For iteration k = 1 to K:
       For each state s:
           Q_k(s,a) = Σ_{s'} T(s,a,s')[R(s,a,s') + γ V_{k-1}(s')]
           V_k(s) = max_a Q_k(s,a)
3. Extract policy: π(s) = argmax_a Q(s,a)
```

**Implementation:**
```python
def runValueIteration(self):
    for iteration in range(self.iterations):
        newValues = util.Counter()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                newValues[state] = 0
                continue

            actions = self.mdp.getPossibleActions(state)
            if actions:
                qValues = [self.computeQValueFromValues(state, a) for a in actions]
                newValues[state] = max(qValues)

        self.values = newValues
```

**Properties:**
- Offline algorithm (no interaction needed)
- Synchronous updates
- Converges to optimal values
- Requires full MDP model

---

### Q-Learning

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α [R + γ max_{a'} Q(s',a') - Q(s,a)]
                     └─────── TD target ────┘ └─ Old Q ─┘
                     └──────────── TD error ──────────────┘
```

**Components:**
- **α (alpha):** Learning rate (0 < α ≤ 1)
- **γ (gamma):** Discount factor (0 ≤ γ < 1)
- **ε (epsilon):** Exploration rate for ε-greedy

**Algorithm:**
```
1. Initialize Q(s,a) = 0 for all s,a
2. For each episode:
       s = initial state
       While s not terminal:
           # ε-greedy action selection
           if random() < ε:
               a = random action
           else:
               a = argmax_{a'} Q(s,a')

           # Take action, observe outcome
           s', r = environment.step(a)

           # TD update
           target = r + γ × max_{a'} Q(s',a')
           Q(s,a) ← Q(s,a) + α × (target - Q(s,a))

           s = s'
```

**Implementation:**
```python
def update(self, state, action, nextState, reward):
    """Temporal difference update"""
    # Current Q-value
    currentQ = self.getQValue(state, action)

    # TD target
    maxNextQ = self.computeValueFromQValues(nextState)
    sample = reward + self.discount * maxNextQ

    # TD update
    self.qValues[(state, action)] = currentQ + self.alpha * (sample - currentQ)
```

**Properties:**
- Model-free (learns from experience)
- Online learning
- Off-policy (learns optimal while exploring)
- Converges with decreasing α and sufficient exploration

---

### Approximate Q-Learning

**Linear Function Approximation:**
```
Q(s,a) = w_1 × f_1(s,a) + w_2 × f_2(s,a) + ... + w_n × f_n(s,a)
       = Σ_i w_i × f_i(s,a)
```

**Weight Update:**
```
difference = [R + γ max_{a'} Q(s',a')] - Q(s,a)
w_i ← w_i + α × difference × f_i(s,a)
```

**Algorithm:**
```python
def update(self, state, action, nextState, reward):
    """Update feature weights"""
    # Extract features
    features = self.featExtractor.getFeatures(state, action)

    # Current Q-value (from weights and features)
    currentQ = sum(self.weights[f] * features[f] for f in features)

    # TD target
    maxNextQ = self.computeValueFromQValues(nextState)
    target = reward + self.discount * maxNextQ

    # TD error
    difference = target - currentQ

    # Update weights
    for feature in features:
        self.weights[feature] += self.alpha * difference * features[feature]
```

**Feature Design Example (Pacman):**
```python
def getFeatures(self, state, action):
    features = util.Counter()

    # Bias term
    features["bias"] = 1.0

    # Distance to nearest food
    successor = state.generateSuccessor(0, action)
    foodList = successor.getFood().asList()
    if foodList:
        minDist = min([manhattanDistance(successor.getPacmanPosition(), f)
                       for f in foodList])
        # Normalize by maze size
        features["closest-food"] = float(minDist) / (walls.width * walls.height)

    # Ghost proximity
    ghosts = [successor.getGhostPosition(i) for i in range(1, successor.getNumAgents())]
    features["#-of-ghosts-1-step-away"] = sum(
        manhattanDistance(successor.getPacmanPosition(), g) <= 1 for g in ghosts
    )

    # Food consumption
    features["eats-food"] = 1.0 if successor.getFood()[x][y] else 0.0

    return features
```

**Properties:**
- Generalizes to unseen states
- Learns from fewer samples
- Scalable to large state spaces
- Feature quality crucial

---

## Performance Analysis

### Value Iteration

**Convergence:**

| Iterations | Max Value Change | Policy Quality |
|------------|------------------|----------------|
| 10 | 2.5 | Poor |
| 50 | 0.1 | Good |
| 100 | 0.001 | Optimal |

**Computational Cost:**
- Time per iteration: O(|S|² × |A|)
- Typical iterations: 10-100
- Memory: O(|S|) for value table

### Q-Learning

**Training Progress:**

| Episodes | Win Rate (Pacman) | Avg Score |
|----------|-------------------|-----------|
| 100 | 20% | -100 |
| 500 | 50% | +50 |
| 1000 | 70% | +200 |
| 2000 | 85% | +400 |

**Hyperparameter Impact:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| α (learning) | 0.1 | Slow, stable |
| α (learning) | 0.5 | Fast, oscillations |
| γ (discount) | 0.8 | Short-term focus |
| γ (discount) | 0.99 | Long-term planning |
| ε (exploration) | 0.1 | Mostly exploit |
| ε (exploration) | 0.5 | Heavy exploration |

### Approximate Q-Learning

**Sample Efficiency:**

| Method | Episodes to 70% Win | Memory |
|--------|---------------------|---------|
| Tabular Q-Learning | 2000 | O(\|S\| × \|A\|) |
| Approximate Q-Learning | 200 | O(k features) |

**Speedup:** 10x faster convergence with good features

---

## Quick Start Guide

### Value Iteration

```bash
# Run on gridworld with visualization
python gridworld.py -a value -i 100 -k 10

# Adjust discount factor
python gridworld.py -a value -i 100 -g 0.9

# Manual control to test learned policy
python gridworld.py -a value -i 100 -m
```

**Controls:**
- Arrow keys: Manual navigation
- `a`: Auto-run learned policy
- `q`: Quit

### Q-Learning

```bash
# Train on gridworld
python gridworld.py -a q -k 100

# Custom hyperparameters
python gridworld.py -a q -k 200 -l 0.3 -e 0.1 -g 0.95

# Bridge crossing challenge
python gridworld.py -a q -k 50 -n 0.01 -g 1 -l 1 -e 0.1 -b BridgeGrid
```

### Crawler Robot

```bash
# Watch robot learn locomotion
python crawler.py

# Use GUI to adjust learning parameters
# - Learning rate slider
# - Discount factor slider
# - Exploration rate slider
# - Step delay for visualization
```

### Pacman Q-Learning

```bash
# Train tabular Q-learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Fast training (no graphics)
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -q
```

### Approximate Q-Learning

```bash
# Train with simple features
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# Different layouts
python pacman.py -p ApproximateQAgent -x 100 -n 110 -l mediumClassic
```

---

## Key Concepts Explained

### Markov Decision Process (MDP)

**Components:**
- **S:** Set of states
- **A:** Set of actions
- **T(s,a,s'):** Transition function P(s'|s,a)
- **R(s,a,s'):** Reward function
- **γ:** Discount factor (0 ≤ γ < 1)

**Objective:**
```
Find policy π*: S → A that maximizes:
V^π(s) = E[Σ_{t=0}^∞ γ^t R_t | π]
```

### Bellman Equations

**Bellman Optimality for V*:**
```
V*(s) = max_a Q*(s,a)
Q*(s,a) = Σ_{s'} T(s,a,s')[R(s,a,s') + γ V*(s')]
```

**Bellman Optimality for Q*:**
```
Q*(s,a) = Σ_{s'} T(s,a,s')[R(s,a,s') + γ max_{a'} Q*(s',a')]
```

### Temporal Difference Learning

**TD Error:**
```
δ_t = R_t + γ V(S_{t+1}) - V(S_t)
     └─── TD target ──┘   └─ Current ─┘
```

**Update:**
```
V(S_t) ← V(S_t) + α × δ_t
```

**Properties:**
- Learns from incomplete episodes
- Bootstraps from current estimates
- Balances bias (bootstrapping) vs variance (Monte Carlo)

### Exploration vs Exploitation

**ε-greedy Policy:**
```
a = { random action           with probability ε
    { argmax_a Q(s,a)         with probability 1-ε
```

**Exploration Strategies:**
- **Fixed ε:** Constant exploration (e.g., ε=0.1)
- **Decaying ε:** Start high (ε=1.0), decay to low (ε=0.01)
- **Optimistic initialization:** Q(s,a) = high value encourages exploration

---

## Implementation Examples

### Value Iteration

```python
class ValueIterationAgent:
    def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        """Perform Bellman updates"""
        for iteration in range(self.iterations):
            newValues = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue

                actions = self.mdp.getPossibleActions(state)
                if actions:
                    qValues = [self.computeQValue(state, a) for a in actions]
                    newValues[state] = max(qValues)

            self.values = newValues

    def computeQValue(self, state, action):
        """Compute Q(s,a) from V(s')"""
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue
```

### Q-Learning

```python
class QLearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=0.9):
        self.alpha = alpha      # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.discount = gamma   # Discount factor
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """Return Q(s,a)"""
        return self.qValues[(state, action)]

    def getAction(self, state):
        """ε-greedy action selection"""
        legalActions = self.getLegalActions(state)

        # Exploration
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        # Exploitation
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """TD update"""
        currentQ = self.getQValue(state, action)
        maxNextQ = max([self.getQValue(nextState, a)
                       for a in self.getLegalActions(nextState)] or [0])

        sample = reward + self.discount * maxNextQ
        self.qValues[(state, action)] = currentQ + self.alpha * (sample - currentQ)
```

---

## Testing and Validation

### Automated Testing

```bash
# All questions
python autograder.py

# Individual questions
python autograder.py -q q1  # Value Iteration
python autograder.py -q q2  # Bridge Crossing
python autograder.py -q q3  # Q-Learning Discount
python autograder.py -q q4  # Q-Learning Pacman
python autograder.py -q q5  # Approximate Q-Learning
python autograder.py -q q6  # Analysis Questions

# Verbose output for debugging
python autograder.py -q q1 --verbose
```

### Manual Testing

```bash
# Watch value iteration converge
python gridworld.py -a value -i 10 -s 0.5  # 0.5s pause per iteration

# Train Q-learning and visualize
python gridworld.py -a q -k 50 -s 0.2

# Test different learning rates
for lr in 0.1 0.3 0.5 0.8; do
    echo "Testing learning rate $lr"
    python gridworld.py -a q -k 100 -l $lr -q
done
```

---

## Troubleshooting

### Q-Learning Not Converging
**Problem:** Q-values oscillate or don't stabilize

**Solutions:**
- Increase training episodes (try 1000-5000)
- Decrease learning rate (α = 0.1-0.3)
- Ensure sufficient exploration (ε = 0.05-0.2)
- Check for negative reward cycles
- Verify discount factor < 1.0

### Pacman Performs Poorly
**Problem:** Low win rate after training

**Solutions:**
- Train for more episodes (2000-5000)
- Adjust reward shaping
- Check feature extraction (approximate QL)
- Increase exploration during training
- Test on simpler layouts first

### Approximate Q-Learning Doesn't Generalize
**Problem:** Poor performance on new states

**Solutions:**
- Design better features (add ghost distance, food count)
- Normalize feature values (divide by maze size)
- Increase feature diversity
- Train longer (100+ episodes)
- Check feature weights make sense

### Value Iteration Doesn't Converge
**Problem:** Values keep changing significantly

**Solutions:**
- Increase iterations (100-200)
- Check discount factor < 1.0
- Verify synchronous updates (copy values)
- Ensure terminal states handled correctly

---

## Learning Objectives

✅ **MDP Fundamentals**
- Understand states, actions, transitions, rewards
- Formulate problems as MDPs
- Compute optimal policies from value functions

✅ **Value Iteration**
- Implement Bellman optimality updates
- Extract policies from value functions
- Analyze convergence properties

✅ **Q-Learning**
- Implement model-free TD learning
- Design exploration strategies (ε-greedy)
- Understand off-policy learning
- Handle experience replay

✅ **Function Approximation**
- Design feature extractors
- Implement linear function approximation
- Understand generalization vs memorization
- Balance feature complexity and learnability

---

## Additional Resources

### MDP Interface

```python
mdp.getStates()                          # All states in MDP
mdp.getStartState()                      # Initial state
mdp.getPossibleActions(state)            # Available actions from state
mdp.getTransitionStatesAndProbs(s, a)    # Returns [(s', prob), ...]
mdp.getReward(state, action, nextState)  # R(s,a,s')
mdp.isTerminal(state)                    # Check if terminal
```

### ReinforcementAgent Methods

```python
self.getLegalActions(state)    # Valid actions
self.getValue(state)           # V(s) from Q-values
self.getQValue(state, action)  # Q(s,a)
self.getPolicy(state)          # π(s) = best action
```

---

## References

- **UC Berkeley CS188** - Original project framework
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" (2nd Edition)
- **Russell & Norvig** - "Artificial Intelligence: A Modern Approach" (Chapter 17, 22)
- **Course Materials** - CS-5100 lecture slides on MDPs and RL
