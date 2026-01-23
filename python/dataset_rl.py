import torch

class TSPDatasetRL(torch.utils.data.Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            first = f.readline().split()
            self.num_instances = int(first[0])
            self.n = int(first[1])

            self.data = []
            for _ in range(self.num_instances):
                points = [list(map(float, f.readline().split())) for _ in range(self.n)]
                self.data.append(torch.tensor(points, dtype=torch.float))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
