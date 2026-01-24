import matplotlib.pyplot as plt
import torch

def plot_single(ax, points, tour, title):
    pts = points[tour]
    pts = torch.cat([pts, pts[:1]], dim=0)

    ax.scatter(points[:, 0], points[:, 1], s=50, c="black", zorder=5)
    ax.plot(pts[:, 0], pts[:, 1], linewidth=2)

    for i, (x, y) in enumerate(points):
        ax.text(x, y, str(i), fontsize=9)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)

def plot_three_tours(points, tour1, tour2, tour3,
                     title1="Optimal",
                     title2="Model",
                     title3="2-Opt"):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_single(axes[0], points, tour1, title1)
    plot_single(axes[1], points, tour2, title2)
    plot_single(axes[2], points, tour3, title3)

    plt.tight_layout()
    plt.show()
