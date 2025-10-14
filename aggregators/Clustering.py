import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

class SimpleClusteringFedAvg(FedAvg):
    def __init__(self, clustering="DBSCAN", **kwargs):
        super().__init__(**kwargs)
        self.clustering = clustering

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Convert FitRes parameters to NumPy arrays
        ndarrays_list = []
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            ndarrays_list.append(np.concatenate([w.ravel() for w in ndarrays]))

        weights = np.array(ndarrays_list)

        # Clustering
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(weights, quantile=0.5, n_samples=min(50, len(weights)))
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
        else:  # DBSCAN
            clusterer = DBSCAN(eps=0.1, min_samples=2)

        clusterer.fit(weights)
        labels = clusterer.labels_
        print(labels)
        # Chooses biggest cluster (Majority)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove outliers

        if not unique_labels:
            # Fallback to normal FedAvg
            return super().aggregate_fit(rnd, results, failures)

        counts = [np.sum(labels == l) for l in unique_labels]
        majority_label = list(unique_labels)[np.argmax(counts)]
        majority_idx = [i for i, lbl in enumerate(labels) if lbl == majority_label]
        print("Majority indices:", majority_idx)

        # Filtering of clients from biggest cluster
        filtered_results = [results[i] for i in majority_idx]

        # Aggregation via FedAvg
        return super().aggregate_fit(rnd, filtered_results, failures)
