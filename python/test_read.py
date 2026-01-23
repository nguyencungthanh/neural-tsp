from tsp_utils import load_dataset, tour_length

data = load_dataset("../data/tsp_train.txt")
print(data.shape)  # (100000, 20, 2)

points = data[0]
tour = list(range(20))
print("Tour length:", tour_length(points, tour))
