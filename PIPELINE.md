# Pipeline

## Option A: n=20 (with supervised pretraining)

### Step 1: Generate data

```bash
cd cpp
g++ -std=c++17 generate_data.cpp -o generate_data

# Raw training data (for supervised labels)
./generate_data 100000 20 ../data/train_raw.txt

# Raw test data
./generate_data 5000 20 ../data/test_raw.txt

# RL training data (no labels needed)
./generate_data 100000 20 ../data/train_rl.txt

# Eval data
./generate_data 5000 20 ../data/eval.txt
```

### Step 2: Compute optimal tours (Held-Karp)

```bash
g++ -std=c++17 make_supervised.cpp -o make_supervised
./make_supervised < ../data/train_raw.txt > ../data/train_supervised.txt
./make_supervised < ../data/test_raw.txt > ../data/test_supervised.txt
```

### Step 3: Supervised pretraining

```bash
cd ../python
python train_supervised.py
```

### Step 4: RL Actor-Critic training

```bash
python train_rl.py
```

### Step 5: Evaluate

```bash
# Neural methods (GPU)
python eval_rl.py

# Classical baselines (CPU)
cd ../cpp
g++ -std=c++17 baselines.cpp -o baselines
./baselines < ../data/eval.txt
```

---

## Option B: n=50 (RL only, no supervised)

### Step 1: Generate data

```bash
cd cpp
g++ -std=c++17 generate_data.cpp -o generate_data
./generate_data 100000 50 ../data/train_rl.txt
./generate_data 5000 50 ../data/eval.txt
```

### Step 2: RL Actor-Critic training

```bash
cd ../python
python train_rl.py
```

### Step 3: Evaluate

```bash
python eval_rl.py

cd ../cpp
g++ -std=c++17 baselines.cpp -o baselines
./baselines < ../data/eval.txt
```

---

## Expected output

```
=== RL pretraining-Greedy ===
Average tour length: X.XXXX

=== RL pretraining-Sampling ===
  T=1.5, N=  128: avg length = X.XXXX
  T=2.0, N=  128: avg length = X.XXXX
  ...

=== RL pretraining-Active Search (first 5 instances) ===
  Instance 1: length = X.XXXX
  ...

=== Active Search from scratch (first 3 instances) ===
  Instance 1: length = X.XXXX
  ...

NN's model average tour: X.XXXXXX
2OPT's model average tour: X.XXXXXX
```

## Reference

| Paper Config | File |
|---|---|
| RL pretraining-Greedy | `eval_rl.py` section 1 |
| RL pretraining-Sampling | `eval_rl.py` section 2 |
| RL pretraining-Active Search | `eval_rl.py` section 3 |
| Active Search from scratch | `eval_rl.py` section 4 |
| Nearest Neighbor baseline | `baselines.cpp` |
| 2-Opt baseline | `baselines.cpp` |
| Optimal (Held-Karp) | `held_karp.cpp` (n≤20 only) |
