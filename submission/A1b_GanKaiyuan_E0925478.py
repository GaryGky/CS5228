import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class MyKMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def initialize_centroids(self, X):

        # Pick the first centroid randomly
        c1 = np.random.choice(X.shape[0], 1)

        # Add first centroids to the list of cluster centers
        self.cluster_centers_ = X[c1]

        # Calculate and add c2, c3, ..., ck (we assume that we always have more unique data points than k!)
        while len(self.cluster_centers_) < self.n_clusters:

            #########################################################################################
            ### Your code starts here ###############################################################
            dist = np.min(euclidean_distances(X, self.cluster_centers_), axis=1)
            prob = dist ** 2 / np.sum(dist ** 2)
            # c is a data point representing the next centroid
            c = X[np.random.choice(X.shape[0], replace=True, p=prob)]
            ### Your code ends here #################################################################
            #########################################################################################                

            # Add next centroid c to the array of already existing centroids
            self.cluster_centers_ = np.concatenate((self.cluster_centers_, [c]), axis=0)

    def assign_clusters(self, X):
        #########################################################################################
        ### Your code starts here ###############################################################

        # Reset all clusters (i.e., the cluster labels)
        self.labels_ = None
        dist = euclidean_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(dist, axis=1)

        ### Your code ends here #################################################################
        #########################################################################################

    def update_centroids(self, X):

        # Initialize list of new centroids with all zeros
        new_cluster_centers_ = np.zeros_like(self.cluster_centers_)

        for cluster_id in range(self.n_clusters):
            new_centroid = None

            #########################################################################################
            ### Your code starts here ###############################################################
            indices = np.where(self.labels_ == cluster_id)
            new_centroid = np.mean(X[indices],axis=0)

            ### Your code ends here #################################################################
            #########################################################################################

            new_cluster_centers_[cluster_id] = new_centroid

            # Check if old and new centroids are identical; if so, we are done
        done = (self.cluster_centers_ == new_cluster_centers_).all()

        # Update lest of centroids
        self.cluster_centers_ = new_cluster_centers_

        # Return TRUE if the centroids have not changeg; return FALSE otherwise
        return done

    def fit(self, X):

        self.initialize_centroids(X)

        self.n_iter_ = 0
        for _ in range(self.max_iter):

            # Update iteration counter
            self.n_iter_ += 1

            # Assign cluster
            self.assign_clusters(X)

            # Update centroids
            done = self.update_centroids(X)

            if done:
                break

        return self
