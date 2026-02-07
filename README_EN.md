# AI Challenge 2025: Reinforcement Learning for Ising Model Optimization

Solution for the **"RL for Ising Model"** challenge by **Rosatom** from team **f0rtYtwO** in the **AI Challenge 2025** competition.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Problem Description

Develop an RL agent capable of minimizing the total energy of a system by sequentially changing vertex charges in a graph from the Ising model.

### Formal Statement

- Graph with 50 vertices
- Each vertex has a charge: $s_i \in \{-1, +1\}$
- Each edge has a charge: $J_{ij} \in \{-1, +1\}$
- Total system energy:

$$
E(s) = - \sum_{(i,j) \in E} J_{ij} \cdot s_i \cdot s_j
$$

**Goal**: Minimize energy through sequential vertex charge modifications.

## Solution

### Architecture

We used **Deep Q-Network (DQN)** with a graph neural network based on **Graph Attention Network (GAT)**.

**Key Components:**

1. **Graph Encoding**: Adjacency matrix and vertex charge vector
2. **GAT Layers**: 3 layers with attention mechanism for graph structure processing
3. **DQN**: Q-value estimation for all possible actions
4. **Experience Replay**: For training stabilization
5. **Target Network**: To reduce correlation in updates

### Results

- **Top score on Round 1 leaderboard**
- Effective energy minimization on test configurations
- Agent learns to find near-optimal solutions

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Running

Open the notebook in Google Colab or Jupyter:

```bash
jupyter notebook AIC_2025_Round_1_Solution.ipynb
```

Main notebook sections:

1. **Problem Formulation** - Mathematical formulation in RL terms
2. **Environment** - Ising model environment implementation
3. **Agent (DQN + GAT)** - Neural network architecture
4. **Training** - Agent training process
5. **Validation** - Testing on validation configurations
6. **Visualization** - Optimization process animation

## Visualization

The notebook includes an interactive animation showing:
- System energy changes over time
- Current graph configuration
- Agent actions at each step

## Methodology

### RL Formalization

- **State**: Graph configuration (adjacency matrix + charge vector)
- **Action**: Vertex selection and charge inversion ($s_i \leftarrow -s_i$)
- **Reward**: Energy change $r = E(s_{\text{before}}) - E(s_{\text{after}})$
- **Policy**: Neural network $\pi_\theta(a \mid s)$

### Network Architecture

```
Input (Graph) → GAT Layer 1 → GAT Layer 2 → GAT Layer 3 → FC Layers → Q-values (50)
```

## 🏗️ Repository Structure

```
├── README.md                          # This file
├── README_RU.md                       # Russian version
├── AIC_2025_Round_1_Solution.ipynb   # Main notebook
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License
└── .gitignore                         # Git ignore rules
```

## 👥 Team

**f0rtYtwO**

## Notes

Despite achieving the top score on the Round 1 leaderboard, the team did not advance further in the competition. Nevertheless, this solution demonstrates effective application of Reinforcement Learning methods to combinatorial optimization problems.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AI Challenge 2025 organizers
- Rosatom for the interesting problem

## Contact

Feel free to reach out if you have questions about the implementation!

---

**Keywords**: reinforcement learning, deep learning, ising model, graph neural networks, DQN, GAT, optimization, PyTorch, AI challenge
