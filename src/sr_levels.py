# sr_levels.py

import numpy as np
import warnings
import math
from sklearn.cluster import DBSCAN

def cluster_pivots(pivot_high_values, pivot_high_weights,
                   pivot_low_values, pivot_low_weights,
                   abs_eps=0.5, min_samples=1):
    """
    Clusters pivot points strictly by percentage difference, using a custom
    pairwise distance matrix (absolute log difference). If two prices differ 
    by less than 'abs_eps'% in ratio terms, they'll likely cluster together.

    Parameters:
      pivot_high_values, pivot_high_weights,
      pivot_low_values,  pivot_low_weights
        - arrays of pivot prices + their weights, from main.py
      abs_eps (float)
        - If abs_eps=0.5 => 0.5%, so eps in log space = ln(1.005)
      min_samples (int)
        - DBSCAN parameter controlling minimum cluster size

    Returns:
      sr_levels (list of floats)
      sr_counts (list of floats)
    """

    # Combine all pivots and weights
    all_values = np.concatenate((pivot_high_values, pivot_low_values))
    all_weights = np.concatenate((pivot_high_weights, pivot_low_weights))

    # No pivots => no SR lines
    if len(all_values) == 0:
        return [], []

    # Remove any non-positive pivot prices (log not defined for <=0)
    if np.any(all_values <= 0):
        warnings.warn("Pivot prices contain non-positive values. Excluding them from clustering.")
        mask = all_values > 0
        all_values = all_values[mask]
        all_weights = all_weights[mask]
        if len(all_values) == 0:
            return [], []

    # ---------------------------------------------------
    # 1) Build the NxN distance matrix using absolute log-differences
    #    distance(x, y) = | log(x) - log(y) |
    # ---------------------------------------------------
    def ratio_distance(vals):
        N = len(vals)
        dist = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i+1, N):
                d = abs(math.log(vals[i]) - math.log(vals[j]))
                dist[i, j] = d
                dist[j, i] = d
        return dist

    dist_matrix = ratio_distance(all_values)

    # ---------------------------------------------------
    # 2) Compute DBSCAN eps in log space:
    #    If abs_eps=0.5 => 0.5% => log(1 + 0.5/100) = log(1.005)
    # ---------------------------------------------------
    eps_log = math.log(1 + abs_eps / 100.0)

    # ---------------------------------------------------
    # 3) Run DBSCAN with metric='precomputed'
    # ---------------------------------------------------
    dbscan = DBSCAN(eps=eps_log, min_samples=min_samples, metric='precomputed')
    dbscan.fit(dist_matrix)
    labels = dbscan.labels_
    unique_labels = set(labels)

    sr_levels = []
    sr_counts = []

    for label in unique_labels:
        if label == -1:  
            # Noise => skip
            continue

        cluster_indices = (labels == label)
        cluster_vals = all_values[cluster_indices]
        cluster_wts  = all_weights[cluster_indices]

        # Weighted mean in log-space (geometric mean), or fallback
        wt_sum = np.sum(cluster_wts)
        if wt_sum == 0:
            # If no weight, do arithmetic mean in log
            cluster_mean_log = np.mean([math.log(v) for v in cluster_vals])
            cluster_weight = len(cluster_vals)
        else:
            cluster_mean_log = np.sum(
                [math.log(cluster_vals[i]) * cluster_wts[i] 
                 for i in range(len(cluster_vals))]
            ) / wt_sum
            cluster_weight = wt_sum

        cluster_mean = math.exp(cluster_mean_log)
        sr_levels.append(cluster_mean)
        sr_counts.append(cluster_weight)

    # Sort ascending
    if sr_levels:
        sort_idx = np.argsort(sr_levels)
        sr_levels  = [sr_levels[i] for i in sort_idx]
        sr_counts  = [sr_counts[i] for i in sort_idx]

    return sr_levels, sr_counts