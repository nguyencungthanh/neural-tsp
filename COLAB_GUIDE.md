# Google Colab + Google Drive Setup Guide

## 1. Upload project to Google Drive

```
My Drive/
└── neural-tsp/
    ├── cpp/
    │   ├── generate_data.cpp
    │   ├── make_supervised.cpp
    │   ├── held_karp.cpp
    │   └── baselines.cpp
    ├── python/
    │   ├── model.py
    │   ├── critic.py
    │   ├── train_supervised.py
    │   ├── train_rl.py
    │   ├── eval_rl.py
    │   ├── active_search.py
    │   ├── search.py
    │   ├── eval_utils.py
    │   ├── dataset.py
    │   └── dataset_rl.py
    └── data/           (empty, will be generated on Colab)
```

Zip your project locally, upload to Google Drive, then unzip in Colab.

---

## 2. Colab Notebook — n=20

Open a new Colab notebook, select **T4 GPU** runtime, then run these cells one by one.

### Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Copy project and install dependencies

```python
# Copy project to Colab local storage (faster than reading from Drive)
!cp -r /content/drive/MyDrive/neural-tsp /content/neural-tsp
%cd /content/neural-tsp

!pip install torch numpy matplotlib
```

### Cell 3: Generate data (C++ on Colab CPU)

```bash
%%bash
cd /content/neural-tsp/cpp
g++ -std=c++17 -O2 generate_data.cpp -o generate_data

./generate_data 100000 20 ../data/train_raw.txt
./generate_data 5000 20 ../data/test_raw.txt
./generate_data 100000 20 ../data/train_rl.txt
./generate_data 5000 20 ../data/eval.txt
```

### Cell 4: Save generated data to Drive

```python
import shutil

shutil.copytree('/content/neural-tsp/data', '/content/drive/MyDrive/neural-tsp/data', dirs_exist_ok=True)
print("Data saved to Google Drive.")
```

### Cell 5: Compute optimal tours with Held-Karp

```bash
%%bash
cd /content/neural-tsp/cpp
g++ -std=c++17 -O2 make_supervised.cpp -o make_supervised

./make_supervised < ../data/train_raw.txt > ../data/train_supervised.txt
./make_supervised < ../data/test_raw.txt > ../data/test_supervised.txt
```

> **Note:** For n=20, 100k instances takes ~5-10 minutes on Colab CPU. For n=50, Held-Karp is infeasible — skip to Cell 8.

### Cell 6: Save supervised data to Drive

```python
import shutil

for f in ['train_supervised.txt', 'test_supervised.txt']:
    shutil.copy(f'/content/neural-tsp/data/{f}', f'/content/drive/MyDrive/neural-tsp/data/{f}')
print("Supervised data saved to Google Drive.")
```

### Cell 7: Supervised pretraining

```python
%cd /content/neural-tsp/python
%run train_supervised.py
```

### Cell 8: Save supervised model to Drive

```python
import shutil

shutil.copy('/content/neural-tsp/python/model.pt', '/content/drive/MyDrive/neural-tsp/python/model.pt')
print("Supervised model saved to Google Drive.")
```

### Cell 9: RL Actor-Critic training

```python
%cd /content/neural-tsp/python
%run train_rl.py
```

### Cell 10: Save RL models and checkpoints to Drive

```python
import shutil
import os

# Save final models
for f in ['actor.pt', 'critic.pt']:
    src = f'/content/neural-tsp/python/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/drive/MyDrive/neural-tsp/python/{f}')

# Save all checkpoints
ckpt_dir = '/content/neural-tsp/python/checkpoints'
if os.path.exists(ckpt_dir):
    shutil.copytree(ckpt_dir, '/content/drive/MyDrive/neural-tsp/python/checkpoints', dirs_exist_ok=True)

print("All models and checkpoints saved to Google Drive.")
```

### Cell 11: Evaluate neural methods

```python
%cd /content/neural-tsp/python
%run eval_rl.py
```

### Cell 12: Evaluate classical baselines

```bash
%%bash
cd /content/neural-tsp/cpp
g++ -std=c++17 -O2 baselines.cpp -o baselines
./baselines < ../data/eval.txt
```

### Cell 13: Save everything to Drive (final backup)

```python
import shutil

# Save all Python outputs
for f in ['actor.pt', 'critic.pt', 'model.pt']:
    src = f'/content/neural-tsp/python/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/drive/MyDrive/neural-tsp/python/{f}')

# Save checkpoints
ckpt_dir = '/content/neural-tsp/python/checkpoints'
if os.path.exists(ckpt_dir):
    shutil.copytree(ckpt_dir, '/content/drive/MyDrive/neural-tsp/python/checkpoints', dirs_exist_ok=True)

# Save generated data
shutil.copytree('/content/neural-tsp/data', '/content/drive/MyDrive/neural-tsp/data', dirs_exist_ok=True)

print("Everything saved to Google Drive.")
```

---

## 3. Colab Notebook — n=50 (RL only)

Same as above, but **skip Cell 5, Cell 6, Cell 7, Cell 8** (Held-Karp is infeasible for n=50).

### Replace Cell 3 with:

```bash
%%bash
cd /content/neural-tsp/cpp
g++ -std=c++17 -O2 generate_data.cpp -o generate_data

./generate_data 100000 50 ../data/train_rl.txt
./generate_data 5000 50 ../data/eval.txt
```

### Then go directly to Cell 9 (RL training).

---

## 4. Resume from checkpoint (if Colab disconnects)

If your Colab session disconnects during training, you can resume from the last checkpoint:

### Resume Cell: Reload and continue RL training

```python
%cd /content/neural-tsp/python

from google.colab import drive
drive.mount('/content/drive')

# Copy saved models back from Drive
!cp /content/drive/MyDrive/neural-tsp/python/actor.pt .
!cp /content/drive/MyDrive/neural-tsp/python/critic.pt .

# Check available checkpoints
import os
if os.path.exists('checkpoints'):
    ckpts = sorted(os.listdir('checkpoints'))
    print(f"Available checkpoints: {ckpts}")
    latest = ckpts[-1]
    print(f"Latest: {latest}")
```

Then edit `train_rl.py` to load from the checkpoint and continue training.

---

## 5. Tips

- **Always work from `/content/neural-tsp/`** — Colab local storage is faster than reading from Drive
- **Save to Drive after every major step** — Colab resets local storage when session ends
- **Checkpoints are saved every 5 epochs** — protects against disconnection during long training
- **Colab session timeout**: ~12 hours. The full n=20 pipeline fits in one session
- **Check GPU**: Run `!nvidia-smi` to confirm T4 is active
- **`%%bash` cells**: Use for all C++ compilation and execution
- **Python cells**: Use `%run` or direct imports for training/eval scripts

## 6. Time estimates on Colab T4

| Step | Time (n=20) | Time (n=50) |
|---|---|---|
| Generate data | ~30 sec | ~30 sec |
| Held-Karp | ~5-10 min | N/A (infeasible) |
| Supervised training | ~5 min | N/A |
| RL training | ~5-10 min | ~45-80 min |
| Eval (all methods) | ~15-30 min | ~30-60 min |
| **Total** | **~30-60 min** | **~1.5-2.5 hours** |
