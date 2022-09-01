import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances


def get_noise_dbscan(X, eps=0.0, min_samples=0):
    euc_mat = euclidean_distances(X, X)
    labels = [0] * len(euc_mat)
    c = 1

    for point in range(0, len(X)):
        if not (labels[point] == 0):
            continue
        neighbors = region_query(euc_mat, point, eps)
        if len(neighbors) < min_samples:
            labels[point] = -1
        else:
            grow_cluster(euc_mat, labels, point, neighbors, c, eps, min_samples)

    core_point_indices, noise_point_indices = [], []
    for i in range(len(labels)):
        if labels[i] == -1:
            noise_point_indices.append(i)
        else:
            core_point_indices.append(i)
    return core_point_indices, noise_point_indices


def grow_cluster(data, labels, point, neighbor_pts, c, eps, min_pts):
    labels[point] = c
    i = 0
    while i < len(neighbor_pts):
        pn = neighbor_pts[i]
        if labels[pn] == -1:
            labels[pn] = c
        elif labels[pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[pn] = c
            pn_neighbor_pts = region_query(data, pn, eps)
            if len(pn_neighbor_pts) >= min_pts:
                neighbor_pts = neighbor_pts + pn_neighbor_pts
        i += 1


def region_query(data_mat, index, eps):
    neighbours = []
    for pn in range(0, len(data_mat)):
        if data_mat[index][pn] < eps:
            neighbours.append(pn)
    return neighbours


if __name__ == '__main__':
    X_dbscan_toy = pd.read_csv('data/a1-dbscan-toy-dataset.txt', header=None, sep=' ').to_numpy()

    my_core_point_indices, my_noise_point_indices = get_noise_dbscan(X_dbscan_toy, eps=0.1, min_samples=10)

    print('Total number of core points: {}\n'.format(len(my_core_point_indices)))
    print('The first 25 indices of the points labeled as core points:\n{}\n'.format(sorted(my_core_point_indices)[:20]))

    print('Total number of noise points: {}\n'.format(len(my_noise_point_indices)))
    print('The indices of all points labeled as noise points:\n{}'.format(sorted(my_noise_point_indices)))
