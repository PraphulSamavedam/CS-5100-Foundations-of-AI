# CS-5100 Foundations of AI

This repository contains projects developed for CS-5100 Foundations of Artificial Intelligence course. Each project explores fundamental AI concepts through practical implementations in the Pacman game environment.

📚 **[View Website](https://praphulsamavedam.github.io/CS-5100-Foundations-of-AI/)**

## Projects

### 1. Project 0: Search Algorithms

A pathfinding system that implements classic uninformed and informed search algorithms to navigate Pacman through mazes to reach goals efficiently.

**Key Approach:** Implements graph search algorithms with various frontier management strategies (Stack, Queue, Priority Queue) and heuristic functions to guide search toward goals while minimizing computational cost.

**✅ Benefits:**
- **Completeness & Optimality**: BFS and A* guarantee shortest path solutions
- **Efficiency**: A* with good heuristics expands 2-3x fewer nodes than uninformed search
- **Flexibility**: Multiple algorithms for different constraints (memory, optimality, speed)
- **Heuristic design**: Manhattan distance and custom heuristics for corners/food problems
- **Educational clarity**: Clean implementations demonstrating search fundamentals
- **Scalability**: Handles mazes from tiny (10 nodes) to big (600+ nodes)

**⚠️ Limitations:**
- **Static environments**: Assumes fixed maze layout, no dynamic obstacles
- **Single agent**: Only Pacman moves, no adversarial planning
- **Perfect information**: Complete knowledge of maze structure required
- **Memory intensive**: BFS/A* store all expanded nodes in memory
- **Heuristic quality dependent**: A* performance relies on admissible heuristic design
- **No learning**: Solutions computed from scratch each time, no experience reuse

**Best For:** Pathfinding in known, static environments where optimal solutions are required and computational resources allow graph search (typical for small-to-medium mazes up to a few hundred nodes).

**[→ Detailed README](0_search/README.md)** | **Algorithms:** DFS, BFS, UCS, A* | **Problems:** Position search, Corners, Food collection | **Heuristics:** Manhattan distance, custom admissible heuristics

**Learning Objectives:**
- Understanding state space search fundamentals
- Implementing graph search with frontier and explored sets
- Designing admissible and consistent heuristics
- Analyzing time/space complexity trade-offs
- Comparing uninformed vs informed search strategies

---

### 2. Project 1: Probabilistic Tracking

A probabilistic inference system that tracks hidden ghost positions using noisy sensor observations through Bayesian Networks, exact inference, and particle filtering techniques.

**Key Approach:** Models ghost tracking as a Hidden Markov Model (HMM) where ghost positions are hidden states and noisy distance readings are observations. Implements both exact inference (forward algorithm) and approximate inference (particle filtering) for belief updates.

**✅ Benefits:**
- **Handles uncertainty**: Tracks ghosts despite noisy, incomplete information
- **Exact inference option**: Forward algorithm provides true probability distributions for small state spaces
- **Scalable approximation**: Particle filtering handles large state spaces with O(N) complexity
- **Multi-target tracking**: Joint particle filtering tracks multiple ghosts simultaneously
- **Principled reasoning**: Bayesian inference provides theoretically sound belief updates
- **Real-time performance**: Particle filters run at interactive speeds (30+ FPS with 100-500 particles)

**⚠️ Limitations:**
- **Exact inference scalability**: O(|X|²) per time step limits to ~1000 positions maximum
- **Particle degeneracy**: Can collapse to single hypothesis if resampling not tuned properly
- **Observation model dependency**: Performance relies on accurate sensor model specification
- **Joint state explosion**: Tracking M ghosts requires exponentially more particles
- **No active sensing**: Cannot choose observations to reduce uncertainty
- **Uniform initialization**: Assumes no prior knowledge of ghost starting positions

**Best For:** Real-time tracking of hidden objects with noisy sensors in discrete state spaces where approximate probabilistic reasoning is acceptable (e.g., robot localization, target tracking with radar/sonar).

**[→ Detailed README](1_tracking/README.md)** | **Methods:** Bayesian Networks, Exact Inference, Particle Filtering, Joint Particle Filtering | **Concepts:** Belief propagation, sensor fusion, temporal reasoning

**Learning Objectives:**
- Constructing Bayesian networks for uncertainty representation
- Implementing exact inference with forward algorithm
- Understanding particle filtering and importance sampling
- Managing belief distributions over hidden states
- Handling sensor noise and observation models
- Tracking multiple targets with joint distributions

---

### 3. Project 2: Reinforcement Learning

A reinforcement learning system where agents learn optimal policies through trial-and-error interaction with Gridworld, Crawler robot, and Pacman environments using value-based methods.

**Key Approach:** Implements model-based (Value Iteration) and model-free (Q-Learning) algorithms that learn value functions from experience. Uses temporal difference updates and function approximation to handle complex state spaces.

**✅ Benefits:**
- **No model required**: Q-learning discovers optimal policies without knowing transition/reward functions
- **Online learning**: Learns while acting in environment, no separate planning phase
- **Function approximation**: Approximate Q-learning generalizes to unseen states via features
- **Sample efficient (approximate)**: Linear function approximation learns from fewer samples than tabular
- **Convergence guarantees**: Provably converges to optimal policy with proper parameters
- **Adaptable**: Works in Gridworld (navigation), Crawler (control), Pacman (game playing)

**⚠️ Limitations:**
- **Sample inefficiency (tabular)**: Q-learning requires thousands of episodes to converge
- **Exploration challenge**: ε-greedy can miss optimal actions if epsilon decays too fast
- **Hyperparameter sensitivity**: Learning rate, discount factor, exploration rate require tuning
- **No safety guarantees**: Exploration can lead to catastrophic states during learning
- **Credit assignment**: Temporal difference error propagates slowly through long episodes
- **Feature engineering required**: Approximate QL performance depends on manual feature design

**Best For:** Learning control policies through experience in environments with unknown dynamics, particularly when simulation is cheap and sample collection is feasible (thousands of episodes acceptable).

**[→ Detailed README](2_reinforcement/README.md)** | **Algorithms:** Value Iteration, Q-Learning, Approximate Q-Learning | **Environments:** Gridworld, Crawler, Pacman | **Concepts:** MDP, Bellman equations, TD learning

**Learning Objectives:**
- Understanding Markov Decision Processes (MDPs)
- Implementing value iteration with Bellman updates
- Model-free learning with Q-learning
- Temporal difference error and bootstrapping
- Exploration vs exploitation trade-offs (ε-greedy)
- Function approximation with feature extraction
- Feature weight learning for generalization

---

### 4. Project 3: Multi-Agent Systems

A game-playing system implementing adversarial search algorithms where Pacman must make optimal decisions against intelligent ghost opponents using game tree search.

**Key Approach:** Models Pacman as a two-player zero-sum game using minimax search with alternating MAX (Pacman) and MIN (ghost) layers. Optimizes with alpha-beta pruning and handles uncertain opponents with expectimax.

**✅ Benefits:**
- **Optimal play guarantee**: Minimax finds best strategy against perfect opponents
- **Alpha-beta efficiency**: 2-10x speedup over minimax through branch pruning (typically 3-5x)
- **Handles uncertainty**: Expectimax models suboptimal/random opponents effectively
- **Depth-limited search**: Configurable depth allows trading off optimality for speed
- **Evaluation functions**: Sophisticated heuristics enable good decisions without full search
- **Real-time performance**: 30ms per move at depth 3-4 with good evaluation functions

**⚠️ Limitations:**
- **Exponential complexity**: O(b^d) limits search depth to ~4-5 plies practically
- **Perfect opponent assumption**: Minimax overestimates opponent capability in real games
- **No learning**: Must search from scratch each move, cannot learn from experience
- **Horizon effect**: Depth limit causes short-sighted decisions
- **Evaluation function dependency**: Quality depends heavily on hand-crafted heuristics
- **No coordination**: Each ghost treated independently, no team strategy modeling

**Best For:** Turn-based adversarial games with perfect information where optimal short-term strategy is needed and search depth can be limited to 3-5 moves (chess tactics, game puzzles, competitive scenarios).

**[→ Detailed README](3_multiagent/README.md)** | **Algorithms:** Reflex Agent, Minimax, Alpha-Beta Pruning, Expectimax | **Concepts:** Game trees, adversarial search, evaluation functions, pruning

**Learning Objectives:**
- Implementing minimax algorithm for adversarial games
- Optimizing search with alpha-beta pruning
- Designing effective evaluation functions
- Understanding expectimax for probabilistic opponents
- Analyzing game tree complexity and optimality
- Balancing search depth with evaluation quality

---

## 🛠️ Technology Stack

- **Python 3.6+** - Primary programming language for all projects
- **Custom game engine** - Pacman simulation framework (UC Berkeley)
- **Standard libraries** - util, math, random (no external AI/ML frameworks)
- **Autograder system** - Automated testing and validation

## 📖 Documentation Structure

Each project folder contains:
- **README.md** - Project overview, algorithms, usage examples, performance analysis
- **DEVELOPMENT.md** - Implementation details, debugging tips, testing strategies, code organization
- **Source code** - Python implementations with inline documentation
- **Test cases** - Automated grading suite with expected outputs
- **Layouts** - Multiple maze/game configurations for testing

## 🚀 Quick Start

### Prerequisites
```bash
# Verify Python installation (3.6 or higher required)
python --version
```

### Running Projects

1. **Navigate to project folder:**
   ```bash
   cd 0_search  # or 1_tracking, 2_reinforcement, 3_multiagent
   ```

2. **Run autograder to validate implementations:**
   ```bash
   python autograder.py
   ```

3. **Run specific algorithms interactively:**
   ```bash
   # Project 0: Search - A* pathfinding
   python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

   # Project 1: Tracking - Particle filtering
   python busters.py -p ParticleAgent -l trickyClassic -k 1

   # Project 2: Reinforcement Learning - Q-learning training
   python gridworld.py -a q -k 100 -e 0.1 -l 0.5

   # Project 3: Multi-agent - Alpha-beta game playing
   python pacman.py -p AlphaBetaAgent -l mediumClassic -a depth=3
   ```

## 📊 Course Progression & Learning Outcomes

### Project 0: Search (Weeks 1-3)
**Foundation:** State space representation, systematic exploration
- ✅ Implement DFS, BFS, UCS, A* search algorithms
- ✅ Design admissible heuristics for complex problems
- ✅ Analyze optimality, completeness, time/space complexity
- ✅ Compare uninformed vs informed search strategies

### Project 1: Tracking (Weeks 4-6)
**Foundation:** Probability theory, Bayesian reasoning
- ✅ Construct Bayesian networks for structured uncertainty
- ✅ Implement exact inference (forward algorithm)
- ✅ Understand particle filtering for approximate inference
- ✅ Apply belief updates for temporal reasoning
- ✅ Handle multi-object tracking with joint distributions

### Project 2: Reinforcement Learning (Weeks 7-10)
**Foundation:** Markov Decision Processes, learning from interaction
- ✅ Implement value iteration with Bellman equations
- ✅ Understand model-free Q-learning with TD updates
- ✅ Design feature extractors for function approximation
- ✅ Balance exploration vs exploitation (ε-greedy)
- ✅ Apply RL to control (Crawler) and game playing (Pacman)

### Project 3: Multi-Agent (Weeks 11-14)
**Foundation:** Game theory, adversarial reasoning
- ✅ Implement minimax for optimal adversarial play
- ✅ Optimize search with alpha-beta pruning
- ✅ Model probabilistic opponents with expectimax
- ✅ Design evaluation functions for complex states
- ✅ Understand game tree complexity and horizon effects

## 🎓 Course Information

**Course:** CS-5100 Foundations of Artificial Intelligence
**Institution:** Northeastern University
**Academic Term:** Fall 2024
**Focus Areas:** Search, probabilistic reasoning, learning, game playing

## 📄 Additional Resources

- **Project Paper.pdf** - Comprehensive final project report with experimental results and analysis

## ⚖️ Academic Integrity

This repository is shared as a **learning resource and reference guide** for understanding AI algorithm implementations.

### 📚 Usage Guidelines

**Permitted Use:**
- ✅ Study code to understand algorithm implementations
- ✅ Reference when debugging your own implementations
- ✅ Learn from code structure and design patterns
- ✅ Use as inspiration for documentation and testing strategies

**Academic Integrity:**
- ❌ Do not copy code for your own coursework
- ❌ Do not submit this work as your own
- ❌ Respect your institution's academic honesty policies
- ⚠️ Always cite sources and write your own implementations

**For Students:** This code should guide your learning, not replace it. Understanding comes from implementing algorithms yourself, making mistakes, debugging, and iterating.

**For Educators:** Feel free to reference this as an example of well-documented student work, but please discourage direct copying.

## 🙏 Acknowledgments

### Course Framework
- **UC Berkeley CS188** - Original Pacman AI project framework and game engine
- **John DeNero and Dan Klein** - Project design, autograder development, and educational materials
- **UC Berkeley AI Research Lab** - Educational infrastructure and problem sets

### Course Instruction
- **CS-5100 Course Staff** at Northeastern University for guidance and project specifications
- **Teaching Assistants** for debugging help and conceptual clarifications
- **Fellow Students** for collaboration, discussion, and peer learning

### Theoretical Foundations
- **Stuart Russell & Peter Norvig** - "Artificial Intelligence: A Modern Approach" (AIMA textbook)
- **Richard Sutton & Andrew Barto** - "Reinforcement Learning: An Introduction"
- **Daphne Koller & Nir Friedman** - "Probabilistic Graphical Models"

## 📞 Getting Started

For detailed information on each project:
1. Navigate to the project folder (0_search, 1_tracking, 2_reinforcement, 3_multiagent)
2. Read the **README.md** for algorithm explanations and usage examples
3. Check **DEVELOPMENT.md** for implementation details and debugging guides
4. Run `python autograder.py` to validate your understanding

## 📜 License

This project is part of academic coursework. The Pacman AI framework is educational software developed by UC Berkeley and is used under their educational license terms.

---

**Note:** Each project folder (0_search, 1_tracking, 2_reinforcement, 3_multiagent) contains comprehensive documentation with algorithm explanations, architecture diagrams, usage examples, performance analysis, implementation details, and troubleshooting guides.
