import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances


def get_noise_dbscan(X, eps=0.0, min_samples=0):
    # Identify the indices of all core points: (before exploration)
    neighbours = euclidean_distances(X) < eps
    core_point_indices = np.where(np.sum(neighbours, axis=1) >= min_samples)[0]

    # 2.1 b) Identify the indices of all noise points ==> noise_point_indices
    points_to_corepoints = euclidean_distances(X, X[core_point_indices])
    noise_point_indices = np.where(np.sum(points_to_corepoints < eps, axis=1) == 0)[0]
    return core_point_indices, noise_point_indices


if __name__ == '__main__':
    X_dbscan_toy = pd.read_csv('data/a1-dbscan-toy-dataset.txt', header=None, sep=' ').to_numpy()

    my_core_point_indices, my_noise_point_indices = get_noise_dbscan(X_dbscan_toy, eps=0.1, min_samples=10)

    print('Total number of core points: {}\n'.format(len(my_core_point_indices)))
    print('The first 25 indices of the points labeled as core points:\n{}\n'.format(sorted(my_core_point_indices)[:20]))

    print('Total number of noise points: {}\n'.format(len(my_noise_point_indices)))
    print('The indices of all points labeled as noise points:\n{}'.format(sorted(my_noise_point_indices)))
