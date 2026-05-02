

# --- Option A: separate data file + label file ---

# AFTER — use the full path
DATA_FILE  = "/Users/cepl/ECKM/ECKM code and dataset/flame.txt"
LABEL_FILE = "/Users/cepl/ECKM/ECKM code and dataset/flame_target.txt"

# --- Option B: single combined file (last column = class label) ---
# COMBINED_FILE = "dataset.txt"     # used only when LABEL_FILE = ""

# Delimiter used in your .txt files: None → any whitespace, ',' → CSV-style
DELIMITER = ','

# Apply PCA before clustering? Recommended when features > 2
APPLY_PCA = True
PCA_COMPONENTS = 2               # target dimensions after PCA

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, ConvexHull, distance
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    homogeneity_score,
    completeness_score,
    adjusted_rand_score
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    if LABEL_FILE:                       # Option A: two files
        P      = np.loadtxt(DATA_FILE,  delimiter=DELIMITER)
        y_true = np.loadtxt(LABEL_FILE, delimiter=DELIMITER).astype(int)
        print(f"[Data]   Loaded features from : {DATA_FILE}")
        print(f"[Labels] Loaded labels from   : {LABEL_FILE}")
    else:                                # Option B: combined file
        raw    = np.loadtxt(COMBINED_FILE, delimiter=DELIMITER)
        P      = raw[:, :-1]
        y_true = raw[:,  -1].astype(int)
        print(f"[Data]   Loaded combined file  : {COMBINED_FILE}")

    print(f"         Feature matrix shape  : {P.shape}")
    print(f"         Label vector shape    : {y_true.shape}")
    print(f"         Number of classes     : {len(np.unique(y_true))}")
    return P, y_true


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def CH(P):
    return ConvexHull(P)

def VD(P):
    return Voronoi(P)

def get_voronoi_vertices(vor):
    return vor.vertices

def in_hull(vor_v, hull):
    hull_polygon = Polygon(hull.points[hull.vertices])
    return np.array([v for v in vor_v if hull_polygon.contains(Point(v))])

def circle_vor_in(vor_in, P):
    circles = []
    for center in vor_in:
        r = np.min(distance.cdist([center], P))
        circles.append((center, r))
    return circles

def sort_circle(circles):
    return sorted(circles, key=lambda x: x[1], reverse=True)

def sort_circum(sorted_circles, P, tol=1e-6):
    circum = []
    for center, r in sorted_circles:
        d    = distance.cdist([center], P)[0]
        pts  = P[np.abs(d - r) < tol]
        for p in pts:
            circum.append(tuple(p))
    return circum

def final_circum(circum):
    return np.array(list(dict.fromkeys(circum)))


# ─────────────────────────────────────────────────────────────────────────────
# ECKM ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def ECKM(P, k):
    hull          = CH(P)
    vor           = VD(P)
    vor_v         = get_voronoi_vertices(vor)
    vor_in        = in_hull(vor_v, hull)

    circles       = circle_vor_in(vor_in, P)
    sorted_circles= sort_circle(circles)

    circum        = sort_circum(sorted_circles, P)
    init_centers  = final_circum(circum)[:k]

    km = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=0)
    km.fit(P)
    return km.labels_, km.cluster_centers_, km.n_iter_


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, P_data, y_true, y_pred):
    return {
        "Method"      : name,
        "Silhouette"  : silhouette_score(P_data, y_pred),
        "DB Index"    : davies_bouldin_score(P_data, y_pred),
        "Homogeneity" : homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "ARI"         : adjusted_rand_score(y_true, y_pred),
    }

def calculate_metrics(name, P_data, labels, centers, iterations):
    euclidean_dist = cdist(P_data, centers, metric="euclidean")
    manhattan_dist = cdist(P_data, centers, metric="cityblock")
    avg_euc = np.mean([np.min(euclidean_dist[i]) for i in range(len(P_data))])
    avg_man = np.mean([np.min(manhattan_dist[i]) for i in range(len(P_data))])
    return {
        "Method"          : name,
        "Iterations"      : iterations,
        "Avg Euclidean"   : round(avg_euc, 4),
        "Avg Manhattan"   : round(avg_man, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(P_vis, labels_km, labels_eckm, centers_eckm):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(P_vis[:, 0], P_vis[:, 1], c=labels_km, s=10, cmap="tab10")
    plt.title("K-Means (Random Init)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.subplot(1, 2, 2)
    plt.scatter(P_vis[:, 0], P_vis[:, 1], c=labels_eckm, s=10, cmap="tab10")
    plt.scatter(centers_eckm[:, 0], centers_eckm[:, 1],
                c="red", marker="X", s=150, label="Centers")
    plt.title("ECKMeans")
    plt.xlabel("Component 1")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("=" * 60)
    print("  ECKMeans – Loading Dataset")
    print("=" * 60)
    P, y_true = load_data()

    # 2. Optional PCA
    if APPLY_PCA:
        pca  = PCA(n_components=PCA_COMPONENTS)
        P_vis = pca.fit_transform(P)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"\n[PCA]    Reduced {P.shape[1]}D → {PCA_COMPONENTS}D  "
              f"(explained variance: {explained:.1f}%)")
    else:
        P_vis = P
        if P.shape[1] != 2:
            raise ValueError(
                "APPLY_PCA=False requires 2-D data for Voronoi/ConvexHull. "
                "Set APPLY_PCA=True or supply a 2-column dataset."
            )

    # 3. Clustering
    k = len(np.unique(y_true))
    print(f"\n[Cluster] k = {k}")

    print("\n  Running ECKMeans ...", end=" ", flush=True)
    labels_eckm, centers_eckm, iter_eckm = ECKM(P_vis, k)
    print(f"done  ({iter_eckm} iterations)")

    print("  Running K-Means (random) ...", end=" ", flush=True)
    km_model   = KMeans(n_clusters=k, init="random", n_init=10, random_state=0)
    labels_km  = km_model.fit_predict(P_vis)
    iter_km    = km_model.n_iter_
    centers_km = km_model.cluster_centers_
    print(f"done  ({iter_km} iterations)")

    # 4. Evaluation tables
    print("\n" + "=" * 60)
    print("  Clustering Quality Metrics")
    print("=" * 60)
    quality = pd.DataFrame([
        evaluate("K-Means", P_vis, y_true, labels_km),
        evaluate("ECKMeans", P_vis, y_true, labels_eckm),
    ]).round(4)
    print(quality.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Performance Metrics")
    print("=" * 60)
    perf = pd.DataFrame([
        calculate_metrics("K-Means",  P_vis, labels_km,   centers_km,   iter_km),
        calculate_metrics("ECKMeans", P_vis, labels_eckm, centers_eckm, iter_eckm),
    ])
    print(perf.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Combined Results")
    print("=" * 60)
    combined = pd.merge(quality, perf, on="Method")[
        ["Method", "Iterations", "Silhouette", "DB Index",
         "Completeness", "Homogeneity", "Avg Euclidean", "Avg Manhattan", "ARI"]
    ]
    print(combined.to_string(index=False))

    # 5. Plot
    plot_results(P_vis, labels_km, labels_eckm, centers_eckm)


if __name__ == "__main__":
    main()
