import torch

def tour_length(points, tour):
    rolled = tour.roll(-1, dims=1)
    a = points.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    b = points.gather(1, rolled.unsqueeze(-1).expand(-1, -1, 2))
    return ((a - b) ** 2).sum(-1).sqrt().sum(-1)

