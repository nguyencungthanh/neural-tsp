import numpy as np

def load_dataset(path):
    with open(path, 'r') as f:
        first_line = f.readline().strip().split()
        num_instances, num_points = map(int, first_line)

        data = []
        coords = []

        for line in f:
            x, y = map(float, line.strip().split())
            coords.append([x, y])
            if len(coords) == num_points:
                data.append(coords)
                coords = []

    return np.array(data)  # shape: (num_instances, num_points, 2)

def tour_length(points, tour):
    """
    points: (n, 2)
    tour: list or array of indices, length n
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = points[tour[i]]
        b = points[tour[(i+1) % n]]
        total += np.linalg.norm(a - b)
    return total

def batch_tour_length(batch_points, batch_tours):
    """
    batch_points: (B, n, 2)
    batch_tours:  (B, n)
    """
    B, n, _ = batch_points.shape
    lengths = np.zeros(B)

    for i in range(B):
        lengths[i] = tour_length(batch_points[i], batch_tours[i])

    return lengths

