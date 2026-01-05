# Foundations of AI

This repository contains projects developed for CS-5100 Foundations of Artificial Intelligence course. Each project explores fundamental AI concepts through practical implementations in the Pacman game environment. 
**NOTE: Only the underlying algorithms are coded, not the environment setup they are forked from the UC Berkley's program**

**Institution:** Northeastern University    
**Focus Areas:** Search algorithms, probabilistic reasoning, reinforcement learning, multi-agent systems, game playing    

## Technologies

- **Python 3.6+** - Primary programming language for all projects
- **Pacman Game Engine** - UC Berkeley's educational AI framework
- **Standard libraries** - util, math, random (no external AI/ML frameworks)
- **Autograder system** - Automated testing and validation
- Classic AI algorithms (search, inference, learning)
- Game tree search and adversarial reasoning

---

## Projects Overview

### [Search Algorithms](search-algorithms.md)
Classic uninformed and informed search for pathfinding in maze environments.

**Key Features:** DFS, BFS, UCS, A* search with custom heuristics    
**Concepts:** State space search, frontier management, admissible heuristics, optimality analysis    
**Performance:** A* expands 2-3x fewer nodes than uninformed search    

---

### [Probabilistic Tracking](probabilistic-tracking.md)
Bayesian inference and particle filtering for tracking hidden ghosts with noisy sensors.

**Key Features:** Exact inference, particle filtering, joint distributions, belief propagation    
**Concepts:** Bayesian Networks, Hidden Markov Models, forward algorithm, importance sampling    
**Performance:** Real-time tracking at 30+ FPS with 100-500 particles    

---

### [Reinforcement Learning](reinforcement-learning.md)
Value-based learning methods for discovering optimal policies through experience.

**Key Features:** Value Iteration, Q-Learning, Approximate Q-Learning, epsilon-greedy exploration    
**Concepts:** Markov Decision Processes, Bellman equations, temporal difference learning, function approximation    
**Performance:** Converges to optimal policy with proper hyperparameters    

---

### [Multi-Agent Systems](multi-agent-systems.md)
Adversarial search algorithms for game playing against intelligent opponents.

**Key Features:** Minimax, Alpha-Beta pruning, Expectimax, evaluation functions    
**Concepts:** Game trees, adversarial search, zero-sum games, optimization through pruning    
**Performance:** 2-10x speedup with alpha-beta pruning (typically 3-5x)    

---

## Repository Structure

```
CS-5100-Foundations-of-AI/
├── 0_search/            # Search Algorithms (Project 0)
├── 1_tracking/          # Probabilistic Tracking (Project 1)
├── 2_reinforcement/     # Reinforcement Learning (Project 2)
└── 3_multiagent/        # Multi-Agent Systems (Project 3)
```

Each project directory contains:
- Python source code implementations
- Detailed README with algorithms and usage
- DEVELOPMENT.md with implementation details
- Test cases and autograder
- Multiple layouts/environments for testing

---

## Getting Started

Each project has its own detailed documentation page (use navigation above). Generally:

1. **Prerequisites**: Python 3.6+ with standard libraries
2. **Navigate**: `cd 0_search` (or other project folder)
3. **Test**: `python autograder.py` to validate implementations
4. **Run**: Execute specific algorithms with command-line options
5. **Documentation**: Click on project links above for comprehensive guides

---

## Contact

**Author:** Praphul Samavedam    
**GitHub:** [@PraphulSamavedam](https://github.com/PraphulSamavedam)
