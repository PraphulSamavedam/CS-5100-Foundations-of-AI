# Probabilistic Tracking Project

## Overview

This project implements **probabilistic inference and tracking algorithms** using Bayesian Networks and Hidden Markov Models (HMMs). The system demonstrates how to reason under uncertainty by tracking ghost positions in Pacman using noisy sensor readings, belief updates, and particle filtering techniques.

The implementation covers exact inference, approximate inference through particle filtering, and dynamic tracking of multiple moving targets. Agents learn to hunt ghosts effectively by maintaining probability distributions over possible ghost locations.

## Tracking System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PROBABILISTIC TRACKING SYSTEM                              │
│                       System Workflow Diagram                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │   HIDDEN MARKOV MODEL            │
                    │  - Hidden States (Ghost Position)│
                    │  - Observations (Noisy Distances)│
                    │  - Transition Model P(X_t|X_t-1) │
                    │  - Sensor Model P(e_t|X_t)       │
                    └──────────────┬───────────────────┘
                                   │
        ┌──────────────────────────┼────────────────────────────┐
        │                          │                            │
        ▼                          ▼                            ▼
┌────────────────┐      ┌────────────────┐        ┌────────────────┐
│ EXACT          │      │ PARTICLE       │        │ JOINT          │
│ INFERENCE      │      │ FILTERING      │        │ PARTICLE       │
│                │      │                │        │ FILTERING      │
│ - Forward      │      │ - Sampling     │        │                │
│   algorithm    │      │ - Resampling   │        │ - Multiple     │
│ - Belief state │      │ - Approximate  │        │   ghosts       │
│ - Exact        │      │   tracking     │        │ - Joint dist.  │
└────────┬───────┘      └────────┬───────┘        └────────┬───────┘
         │                       │                         │
         └───────────────────────┼─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   BELIEF UPDATE         │
                    │  P(X_t|e_1:t) =         │
                    │    α P(e_t|X_t)         │
                    │    Σ P(X_t|x_t-1)       │
                    │      P(x_t-1|e_1:t-1)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   GHOST HUNTING         │
                    │  - Choose actions       │
                    │  - Move toward beliefs  │
                    │  - Bust ghosts          │
                    └─────────────────────────┘

Key Algorithms:
├─ Exact Inference: Forward algorithm for belief updates
├─ Particle Filter: Approximate inference via sampling
└─ Joint Particle: Track multiple objects simultaneously
```

## Features

### Core Capabilities

1. **Bayesian Network Construction** - Building graphical models
2. **Exact Inference** - Forward algorithm for exact belief updates
3. **Particle Filtering** - Approximate inference through sampling
4. **Joint Particle Filtering** - Tracking multiple ghosts
5. **Greedy Ghost Hunting** - Action selection based on beliefs

### Inference Methods

#### 1. Exact Inference
- **Method:** Forward algorithm (recursive belief update)
- **Accuracy:** Exact probability distribution
- **Complexity:** O(|X|²) per time step
- **Use Case:** Small state spaces

#### 2. Particle Filtering
- **Method:** Sampling-based approximation
- **Accuracy:** Approximates true distribution
- **Complexity:** O(N) for N particles
- **Use Case:** Large state spaces

#### 3. Joint Particle Filtering
- **Method:** Sample joint distributions
- **Accuracy:** Approximate joint beliefs
- **Complexity:** O(N × M) for M ghosts
- **Use Case:** Multiple target tracking

## Quick Start

### Requirements
- **Python 3.6+** - For running tracking algorithms
- All dependencies included in project files
- No external packages required

### Running Tracking Agents

#### Exact Inference
```bash
python busters.py -p BasicAgentAA -l trickyClassic
```

#### Particle Filtering
```bash
python busters.py -p ParticleAgent -l trickyClassic -k 1
```

#### Joint Particle Filtering
```bash
python busters.py -p JointParticleAgent -l trickyClassic -k 2
```

### Command Line Options

**Parameters:**
- `-p` - Agent type (BasicAgentAA, ParticleAgent, JointParticleAgent)
- `-l` - Layout (trickyClassic, smallClassic, etc.)
- `-k` - Number of ghosts
- `-n` - Number of games
- `-q` - Quiet mode (no graphics)

## Usage

### 1. Bayesian Network Construction

Build a Bayes net representing the ghost tracking problem:

```bash
python autograder.py -q q1
```

**Structure:**
- Nodes: Pacman position, Ghost positions, Observations
- Edges: Dependencies between variables
- CPTs: Conditional probability tables

### 2. Exact Inference

Track single ghost with exact probabilities:

```bash
python busters.py -p BasicAgentAA -l trickyClassic -k 1
```

**Process:**
1. Initialize uniform belief over all positions
2. Observe noisy distance reading
3. Update belief using Bayes' rule
4. Predict ghost movement (time elapse)
5. Repeat observe → update → predict

### 3. Particle Filtering

Approximate tracking with particles:

```bash
python busters.py -p ParticleAgent -l trickyClassic -k 1
```

**Algorithm:**
1. Initialize N particles uniformly
2. For each time step:
   - Predict: Move each particle per transition model
   - Observe: Weight particles by observation likelihood
   - Resample: Draw new particles from weighted distribution

### 4. Joint Particle Filtering

Track multiple ghosts simultaneously:

```bash
python busters.py -p JointParticleAgent -l trickyClassic -k 2
```

**Handling Multiple Targets:**
- Each particle represents joint state (all ghost positions)
- Sample joint distributions
- Update all positions together
- More complex but handles interactions

## How It Works

### Bayesian Network

**Structure:**
```
      Pacman
     /      \
    /        \
Ghost0      Ghost1
    |          |
    |          |
  Obs0       Obs1
```

**Semantics:**
- P(Obs0 | Ghost0, Pacman): Sensor model
- P(Ghost_t | Ghost_t-1): Transition model

### Exact Inference (Forward Algorithm)

**Belief Update:**
```
P(X_t | e_1:t) = α P(e_t | X_t) Σ_{x_t-1} P(X_t | x_t-1) P(x_t-1 | e_1:t-1)
```

**Algorithm:**
```python
def observe(self, observation, gameState):
    """
    Update belief given new observation
    """
    # Get all possible positions
    allPositions = self.legalPositions

    # Update beliefs using Bayes rule
    for pos in allPositions:
        # P(e_t | X_t = pos)
        observationProb = self.getObservationProb(observation, pos, gameState)

        # P(X_t = pos | e_1:t) ∝ P(e_t | X_t = pos) × P(X_t = pos | e_1:t-1)
        self.beliefs[pos] *= observationProb

    # Normalize
    self.beliefs.normalize()

def elapseTime(self, gameState):
    """
    Predict belief after ghost moves
    """
    newBeliefs = util.Counter()

    for oldPos in self.legalPositions:
        # Get transition probabilities from oldPos
        newPosDist = self.getPositionDistribution(gameState, oldPos)

        for newPos, prob in newPosDist.items():
            # P(X_t | e_1:t-1) = Σ P(X_t | x_t-1) P(x_t-1 | e_1:t-1)
            newBeliefs[newPos] += prob * self.beliefs[oldPos]

    self.beliefs = newBeliefs
    self.beliefs.normalize()
```

### Particle Filtering

**Algorithm:**
```python
def initializeParticles(self):
    """Sample N particles uniformly"""
    self.particles = []
    for i in range(self.numParticles):
        self.particles.append(random.choice(self.legalPositions))

def observeUpdate(self, observation, gameState):
    """Weight and resample particles"""
    weights = []

    for particle in self.particles:
        # Weight by observation likelihood
        weight = self.getObservationProb(observation, particle, gameState)
        weights.append(weight)

    # Check for all-zero weights
    if sum(weights) == 0:
        self.initializeParticles()
        return

    # Resample particles based on weights
    self.particles = util.nSample(weights, self.particles, self.numParticles)

def elapseTime(self, gameState):
    """Predict particle positions"""
    newParticles = []

    for oldParticle in self.particles:
        # Sample new position from transition model
        newPosDist = self.getPositionDistribution(gameState, oldParticle)
        newParticle = util.sample(newPosDist)
        newParticles.append(newParticle)

    self.particles = newParticles
```

### Joint Particle Filtering

**Joint State:**
```python
# Each particle is a tuple of all ghost positions
particle = (ghost0_pos, ghost1_pos, ghost2_pos, ...)
```

**Joint Update:**
```python
def observeUpdate(self, observation, gameState):
    """Update joint belief"""
    weights = []

    for particle in self.particles:
        weight = 1.0

        # Multiply observation probabilities for all ghosts
        for i, pos in enumerate(particle):
            obs = observation[i]
            weight *= self.getObservationProb(obs, pos, gameState, i)

        weights.append(weight)

    # Resample joint particles
    self.particles = util.nSample(weights, self.particles, self.numParticles)
```

## Performance Considerations

### Exact Inference
- **Time:** O(|X|² × T) for T time steps
- **Space:** O(|X|) for belief distribution
- **Scalability:** Limited to small state spaces (~1000 positions)

### Particle Filtering
- **Time:** O(N × T) for N particles
- **Space:** O(N)
- **Scalability:** Handles large state spaces
- **Accuracy:** Improves with more particles

### Joint Particle
- **Time:** O(N × M × T) for M ghosts
- **Space:** O(N × M)
- **Scalability:** Exponential in number of targets
- **Accuracy:** Requires more particles than single-target

## Project Structure

```
tracking/
├── inference.py              # Main inference implementations
│   ├── constructBayesNet()
│   ├── ExactInference
│   ├── ParticleFilter
│   └── JointParticleFilter
│
├── bustersAgents.py          # Ghost hunting agents
│   ├── BustersAgent
│   ├── BasicAgentAA
│   ├── ParticleAgent
│   └── JointParticleAgent
│
├── busters.py                # Game engine
├── bayesNet.py               # Bayes net utilities
├── factorOperations.py       # Factor manipulation
│
├── layouts/                  # Maze files
└── autograder.py             # Testing
```

## Testing

```bash
# All tests
python autograder.py

# Individual questions
python autograder.py -q q1  # Bayes Net
python autograder.py -q q2  # Exact Inference
python autograder.py -q q3  # Time Elapse
python autograder.py -q q4  # Particle Filtering
python autograder.py -q q5  # Joint Particle
```

## Troubleshooting

### Beliefs Don't Update
**Problem:** Distribution stays uniform

**Solutions:**
1. Check observation probability calculation
2. Verify normalization
3. Ensure non-zero weights

### Particles Collapse
**Problem:** All particles at same location

**Solutions:**
1. Add particle diversity in initialization
2. Check resampling logic
3. Verify transition model provides spread

### Joint Tracking Fails
**Problem:** Poor performance with multiple ghosts

**Solutions:**
1. Increase number of particles
2. Check joint observation probabilities
3. Verify joint state representation

## License

This project is part of academic coursework for CS-5100 at Northeastern University.

**Attribution:** The Pacman AI projects were developed at UC Berkeley by John DeNero and Dan Klein.

### 📚 Usage as Reference

This repository is intended as a **learning resource and reference guide**. Please respect academic integrity policies at your institution.

## Acknowledgments

- **UC Berkeley CS188** for the Pacman AI framework
- **CS-5100 course staff** for guidance and project specifications
