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
