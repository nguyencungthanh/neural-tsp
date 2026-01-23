import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    def __init__(self, path="../data/tsp_n12_supervised.txt"):
            
        self.points = []
        self.tours = []

        with open("../data/tsp_n12_supervised.txt", 'r') as f:
            num_instances, n = map(int, f.readline().split())

            for _ in range(num_instances):
                pts = []
                for _ in range(n):
                    x, y = map(float, f.readline().split())
                    pts.append([x, y])

                tour = list(map(int, f.readline().split()))

                self.points.append(torch.tensor(pts, dtype=torch.float32))
                self.tours.append(torch.tensor(tour, dtype=torch.long))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.tours[idx]

# ds = TSPDataset("data/train_sup_n12.txt")
# print(ds[0][1])  # in tour đầu tiên
