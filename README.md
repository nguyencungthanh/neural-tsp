# 🧠 Neural Combinatorial Optimization for the Traveling Salesman Problem (TSP)

This project explores learning-based approaches for solving the **Euclidean Traveling Salesman Problem (TSP)** using:
- Supervised Learning (Pointer Networks)
- Reinforcement Learning (Policy Gradient)
- Classical baselines (Nearest Neighbor, 2-Opt, Held–Karp)

The goal is to understand how neural networks can learn heuristics for NP-hard combinatorial optimization problems.

# 📌 Problem Definition

Given n 2D points, the Traveling Salesman Problem asks for the shortest possible tour that visits each point exactly once and returns to the start.

We focus on **Euclidean TSP**, where distances are standard $L_2$ distances.

# 🏗 Project Structure
```graphql
neural-tsp/
│
├── cpp/                # C++ code (fast data + classical solvers)
│   ├── generate_data.cpp
│   ├── held_karp.cpp
│   ├── make_supervised.cpp
│   └── baselines.cpp
│
├── python/             # Deep learning models and training
│   ├── model.py
│   ├── dataset.py   
|   ├── dataset_rl.py
│   ├── train_supervised.py
│   ├── train_rl.py
|   ├── eval_supervised.py
│   └── eval_rl.py
│
├── visualization/      # Visualization comparing Optimal tour, Prediction tour, 2-Opt tour 
|   ├── heuristics.py
│   ├── visualize.py   
|   ├── plot_tours.py
|
├── data/               # Generated datasets
├── requirements.txt
└── README.md
```

# ⚙️ Installation
```bash 
git clone https://github.com/nguyencungthanh/neural-tsp.git
cd neural-tsp
pip install -r requirements.txt
``` 

# 📊 Dataset Generation (C++)

## 1️⃣ Generate raw TSP instances
```css 
<num_instances> <num_points>
x y
x y
...
``` 

```bash
g++ -std=c++17 generate_data.cpp -o generate_data && ./generate_data 10000 12 ../data/tsp_n12_raw.txt   # generate data for supervised learning
g++ -std=c++17 generate_data.cpp -o generate_data && ./generate_data 100000 20 ../data/tsp_train.txt   # generate data for reinforcement learning
g++ -std=c++17 generate_data.cpp -o generate_data && ./generate_data 5000 20 ../data/tsp_eval_20.txt   # generate data for evaluation 
``` 

## 2️⃣ Compute optimal tours (Held–Karp)

Convert raw instances into supervised training data:
```bash 
g++ -std=c++17 make_supervised.cpp -o make_supervised && ./make_supervised < ../data/tsp_n12_raw.txt > ../data/tsp_n12_supervised.txt
``` 

# 🤖 Model: Pointer Network
We use a **Pointer Network** to model permutations. The model learns to output a sequence of city indices representing a tour.

**Inputs:**
[batch size, n, 2] point coordinates

**Outputs:**
A permutation of {0, … ,n-1}

# 🎯 Training
**Supervised Learning (Imitation of Optimal Tours)** 
```bash 
cd python
python train_supervised.py 
``` 
Loss: Cross-entropy over next-city prediction.

--- 
**Reinforcement Learning (Policy Gradient)** 

Fine-tune the supervised model to minimize tour length directly.
```bash 
python train_rl.py
``` 
Reward: Negative tour length 

# 📈 **Evaluation**

Compare the learned model against classical heuristics:
```bash 
python eval_rl.py
g++ -std=c++17 baselines.cpp -o baselines && ./baselines < ../data/tsp_eval_20.txt
```

Metrics: 
- Average tour length
- Optimality gap vs Held–Karp
- Comparison with:
    - Nearest Neighbor
    - 2-Opt

# 📚 References

[1] Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer Networks*. arXiv:1506.03134. https://doi.org/10.48550/arXiv.1506.03134

[2] Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). *Neural combinatorial optimization with reinforcement learning*. arXiv:1611.09940. https://doi.org/10.48550/arXiv.1611.09940