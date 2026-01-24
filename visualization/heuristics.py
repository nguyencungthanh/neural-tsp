import torch
import math

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def nearest_neighbor(points):
    n = points.size(0)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n-1):
        last = tour[-1]
        best_j = None
        best_d = 1e9
        for j in range(n):
            if not visited[j]:
                d = dist(points[last], points[j])
                if d < best_d:
                    best_d = d
                    best_j = j
        tour.append(best_j)
        visited[best_j] = True

    return torch.tensor(tour)

def two_opt(points, tour):
    tour = tour.tolist()
    n = len(tour)
    improved = True

    while improved:
        improved = False
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                a, b = tour[i-1], tour[i]
                c, d = tour[k], tour[(k+1) % n]

                old = dist(points[a], points[b]) + dist(points[c], points[d])
                new = dist(points[a], points[c]) + dist(points[b], points[d])

                if new < old:
                    tour[i:k+1] = reversed(tour[i:k+1])
                    improved = True

    return torch.tensor(tour)
