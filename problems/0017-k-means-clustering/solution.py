import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.asarray(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)   # copy: updated in place below
    if centroids.shape[0] != k:
        raise ValueError("need exactly k initial centroids")

    assignments = None
    for _ in range(max_iterations):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
        new_assignments = dists.argmin(axis=1)
        if assignments is not None and np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        for j in range(k):
            members = points[assignments == j]
            if len(members) > 0:            # empty cluster: keep previous centroid
                centroids[j] = members.mean(axis=0)

    return [tuple(round(float(v), 4) for v in c) for c in centroids]