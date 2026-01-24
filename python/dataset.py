import torch

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
            
        self.points = []
        self.tours = []

        with open(filename) as f:
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

