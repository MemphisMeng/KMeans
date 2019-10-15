import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from PreProcessing import impute_missing_values, date_manipulation, one_hot_encoding, \
    manual_one_hot_encoding, drop_cols, normalize
import argparse
from collections import defaultdict
import time
from sklearn.utils import check_random_state
from math import sqrt


class Kmeans(object):
    def __init__(self, n_clusters, init, max_iter=300, random_state=None, n_init=10, tol=1e-2):
        self.k = n_clusters  # number of clusters
        self.init = init  # way of initialization
        self.max_iter = max_iter  # maximum iterations
        self.random_state = check_random_state(random_state)  # random state
        self.n_init = n_init  # the number of clustering processing times, we select the best one
        self.tol = tol  # threshold of stopping the iteration

    # establishing the model
    def fit(self, dataset):
        self.tol = self._tolerance(dataset, self.tol)

        bestError = None
        bestCenters = None
        bestLabels = None
        for i in range(self.n_init):
            labels, centers, error = self._kmeans(dataset)
            if bestError is None or error < bestError:
                bestError = error
                bestCenters = centers
                bestLabels = labels
        self.centers = bestCenters
        return bestLabels, bestCenters, bestError

    # predict the new data given the well trained model
    def predict(self, X):
        return self.update_labels_error(X, self.centers)

    # a combination of fit() & predict()
    def fit_predict(self, dataset):
        if self.init == '1d':
            pca = PCA(n_components=1)
            dataset = pca.fit_transform(dataset)
        self.fit(dataset)
        return self.predict(dataset)

    # The main thread of Kmeans, describing the whole procedure of a clustering
    def _kmeans(self, dataset):
        self.dataset = dataset
        bestError = None
        bestCenters = None
        bestLabels = None
        centerShiftTotal = 0
        centers = self._init_centroids(dataset)

        for i in range(self.max_iter):
            oldCenters = centers.copy()
            labels, error = self.update_labels_error(dataset, centers)
            centers = self.update_centers(dataset, labels, centers)

            if bestError == None or error < bestError:
                bestLabels = labels.copy()
                bestCenters = centers.copy()
                bestError = error

            # the offsets between the current centroids and the last set of ones
            centerShiftTotal = np.linalg.norm(oldCenters - centers) ** 2
            if centerShiftTotal <= self.tol:
                break

        # We need to update again the label and error if the for loop above outputs a different result
        if centerShiftTotal > 0:
            bestLabels, bestError = self.update_labels_error(dataset, bestCenters)

        return bestLabels, bestCenters, bestError

    # k points are picked at random
    def _init_centroids(self, dataset):
        n_samples = dataset.shape[0]
        centers = []
        if self.init == "random":
            seeds = self.random_state.permutation(n_samples)[:self.k]
            centers = dataset[seeds]
        elif self.init == "k-means++" or self.init == '1d':
            centers = self._init_centroids_kpp(dataset)
        return np.array(centers)

    def _tolerance(self, dataset, tol):
        variances = np.var(dataset, axis=0)
        return np.mean(variances) * tol

    # Update the labels while re-calculating the errors
    def update_labels_error(self, dataset, centers):
        labels = self.assign_points(dataset, centers)
        new_means = defaultdict(list)
        error = 0
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            error += sqrt(np.sum(np.square(points - newCenter)))

        return labels, error

    # Update the center of each cluster
    def update_centers(self, dataset, labels, centers):
        new_means = defaultdict(list)
        newcenters = []
        zero_clusters = []  # contain the index of the clusters where there are zero elements

        for k in range(self.k):
            if k not in labels:
                zero_clusters.append(k)

        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            newcenters.append(newCenter)

        for k in zero_clusters:
            newcenters.append(centers[k])

        return np.array(newcenters)

    # assign each point to the cluster of which the center is the clsest next to itself
    def assign_points(self, dataset, centers):
        labels = []
        for point in dataset:
            shortest = float("inf")  # positive infinity
            shortest_index = 0
            for j in range(len(centers)):
                val = self.distance(point, centers[j])
                if val < shortest:
                    shortest = val
                    shortest_index = j
            labels.append(shortest_index)

        return labels

    def distance(self, a, b):
        dimensions = len(a)
        _sum = 0
        for d in range(dimensions):
            difference_sq = pow(a[d] - b[d], 2)
            _sum += difference_sq
        return pow(_sum, 0.5)

    # The way of the initialization of the center points
    def _init_centroids_kpp(self, dataset):
        n_samples, n_features = dataset.shape
        # n_samples = len(dataset)
        # n_features = len(dataset[0])
        centers = np.empty((self.k, n_features))
        # n_local_trials stands for the number of candidates
        n_local_trials = None
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.k))

        # pick the first random point
        center_id = self.random_state.randint(n_samples)
        centers[0] = dataset[center_id]

        # closest_dist_sq is the closest distance from each data point to the centroid
        closest_dist_sq = self.distance(centers[0, np.newaxis], dataset)
        # current_pot所有最短距离的和
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.k):
            # Project the random address to the n_local_trials
            rand_vals = self.random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)

            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                distance_to_candidate = self.distance(dataset[candidate_ids[trial], np.newaxis], dataset)
                new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidate)
                new_pot = new_dist_sq.sum()

                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = dataset[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-StringA", "--path_to_movies", required=True, help="path_to_movies", type=str)
    ap.add_argument("-ValueB", "--k", required=True, help="number of k", type=int)
    ap.add_argument("-StringC", "--init", required=True, help="initialization", type=str)
    args = vars(ap.parse_args())

    file = pd.read_csv(args["path_to_movies"])
    if args['path_to_movies'].contains('movies'):
        file_2 = normalize(drop_cols(manual_one_hot_encoding(
            one_hot_encoding(date_manipulation(impute_missing_values(file))))))
    warnings.filterwarnings('ignore')
    file_2['budget'].loc[file['budget'].isna()] = 0
    file_2['popularity'].loc[file['popularity'].isna()] = 0
    file_2['revenue'].loc[file['revenue'].isna()] = 0
    file_2['runtime'].loc[file['runtime'].isna()] = 0
    file_2['vote_average'].loc[file['vote_average'].isna()] = 0
    file_2['vote_count'].loc[file['vote_count'].isna()] = 0
    km = Kmeans(args["k"], args["init"])
    startTime = time.time()
    labels, errors = km.fit_predict(file_2.values)
    file_2['label'] = -1
    # generate a new property to distinguish the cluster to which a data point belongs
    for i in range(len(labels)):
        file_2['label'].loc[file.index == i] = labels[i] + 1
    print("Total time: ", time.time() - startTime)
    result = file.merge(file_2, left_index=True, right_index=True, how='right')
    # generate an output file
    result[['id', 'label']].to_csv('output.csv', index=False)


