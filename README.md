# Neural TSP with Reinforcement Learning

This project explores using neural networks (Pointer Networks / Transformers) 
to learn heuristics for solving the Traveling Salesman Problem (TSP).

## Features
- Neural heuristic trained with Reinforcement Learning
- Comparison with classical heuristics (Nearest Neighbor, 2-opt)
- C++ implementations for fast baselines
- Visualization of tours

## Project Structure
- `cpp/` : TSP data generation and baseline solvers
- `python/` : PyTorch models and training scripts
- `notebooks/` : Visualization and analysis

## Setup

```bash
git clone https://github.com/nguyencungthanh/neural-tsp.git
cd neural-tsp
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt


🧠 Neural Combinatorial Optimization for the Traveling Salesman Problem (TSP)

This project explores learning-based approaches for solving the Euclidean Traveling Salesman Problem (TSP) using:

Supervised Learning (Pointer Networks)

Reinforcement Learning (Policy Gradient)

Classical baselines (Nearest Neighbor, 2-Opt, Held–Karp)

The goal is to understand how neural networks can learn heuristics for NP-hard combinatorial optimization problems.

📌 Problem Definition

Given n 2D points, the Traveling Salesman Problem asks for the shortest possible tour that visits each point exactly once and returns to the start.

We focus on Euclidean TSP, where distances are standard L_2 distances.

🏗 Project Structure
```graphql
neural-tsp/
│
├── src/                # C++ code (fast data + classical solvers)
│   ├── generate_data.cpp
│   ├── held_karp.cpp
│   ├── make_supervised.cpp
│   └── baselines.cpp
│
├── python/             # Deep learning models and training
│   ├── model.py
│   ├── dataset.py
│   ├── train_supervised.py
│   ├── train_rl.py
│   └── evaluate.py
│
├── data/               # Generated datasets
├── requirements.txt
└── README.md
```

⚙️ Installation
```
git clone <your-repo-url>
cd neural-tsp
pip install -r requirements.txt
``` 

📊 Dataset Generation (C++)

1️⃣ Generate raw TSP instances
```
<num_instances> <num_points>
x y
x y
...
``` 
```
./generate_data 100000 12 data/train_raw_n12.txt
./generate_data 10000 12 data/test_raw_n12.txt
``` 

2️⃣ Compute optimal tours (Held–Karp)

Convert raw instances into supervised training data:
```
./make_supervised < data/train_raw_n12.txt > data/train_sup_n12.txt
./make_supervised < data/test_raw_n12.txt > data/test_sup_n12.txt
``` 

🤖 Model: Pointer Network
We use a Pointer Network to model permutations.
The model learns to output a sequence of city indices representing a tour.

Inputs:
[batch_size, n, 2] point coordinates

Outputs:
A permutation of {0 … n-1}

🎯 Training
**Supervised Learning (Imitation of Optimal Tours)** 
```
cd python
python train_supervised.py
``` 
Loss: Cross-entropy over next-city prediction.

**Reinforcement Learning (Policy Gradient)** 

Fine-tune the supervised model to minimize tour length directly.
```py
python train_rl.py
``` 
Reward: Negative tour length

📈 **Evaluation**

Compare the learned model against classical heuristics:
```
python evaluate.py
```

Metrics:
- Average tour length
- Optimality gap vs Held–Karp
- Comparison with:
    - Nearest Neighbor
    - 2-Opt