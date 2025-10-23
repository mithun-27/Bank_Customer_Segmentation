import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from .config import PATHS, SETTINGS
from .data_utils import load_raw_dataframe, choose_numeric_features, encode_simple_categoricals
from .visualize import plot_elbow, plot_silhouette, plot_pca_scatter, save_summary

# ---- Tunables for speed/quality tradeoff ----
SIL_SAMPLE_SIZE = 2000  # set to 1000 if you still find it slow; or None for full data (slow!)
N_INIT = SETTINGS.n_init
RANDOM_STATE = SETTINGS.random_state

def run_pipeline():
    os.makedirs(PATHS.data_dir, exist_ok=True)
    os.makedirs(PATHS.reports_dir, exist_ok=True)

    # 1) Load raw data
    raw = load_raw_dataframe()

    # 2) Select features (numeric auto by default)
    X = choose_numeric_features(raw, forced_features=SETTINGS.forced_features)

    # If you want to include simple categoricals (e.g., Gender), uncomment:
    # X = encode_simple_categoricals(pd.concat([X, raw[['Gender']]], axis=1))

    # 3) Impute NaNs
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    pd.DataFrame({"feature": X.columns, "median_imputed": imputer.statistics_}).to_csv(
        os.path.join(PATHS.reports_dir, "imputer_params.csv"), index=False
    )

    # 4) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # 5) K search (elbow + sampled silhouette)
    k_values = list(range(SETTINGS.k_min, SETTINGS.k_max + 1))
    sse, sils = [], []

    # Choose indices for silhouette sampling (if dataset is large)
    if SIL_SAMPLE_SIZE is not None and SIL_SAMPLE_SIZE < len(X_scaled):
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(X_scaled), size=SIL_SAMPLE_SIZE, replace=False)
        X_for_sil = X_scaled[sample_idx]
    else:
        X_for_sil = X_scaled  # use full data (can be slow)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels_full = km.fit_predict(X_scaled)
        sse.append(km.inertia_)
        # silhouette on subset only (labels for subset are taken by refitting on subset)
        # Faster & consistent: fit on subset, compute silhouette on subset
        km_sub = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels_sub = km_sub.fit_predict(X_for_sil)
        sils.append(silhouette_score(X_for_sil, labels_sub, metric="euclidean"))

    elbow_path = plot_elbow(k_values, sse)
    sil_path = plot_silhouette(k_values, sils)

    # 6) Pick best K by max (sampled) silhouette
    best_k = int(k_values[int(np.argmax(sils))])

    # 7) Final fit on FULL data with best_k
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(X_scaled)

    # 8) Centroids (original scale)
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids, columns=X.columns)
    centroid_df.insert(0, "Cluster", range(best_k))
    centroids_csv = os.path.join(PATHS.reports_dir, "centroids.csv")
    centroid_df.to_csv(centroids_csv, index=False)

    # 9) PCA for 2D viz
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    out_df = raw.copy()
    out_df["Cluster"] = labels
    out_df["PCA1"] = X_pca[:, 0]
    out_df["PCA2"] = X_pca[:, 1]

    clustered_csv = os.path.join(PATHS.reports_dir, "data_clustered.csv")
    out_df.to_csv(clustered_csv, index=False)

    pca_png = plot_pca_scatter(out_df)
    summary_md = save_summary(best_k, k_values, sse, sils, centroids_csv, clustered_csv)

    return {
        "best_k": best_k,
        "elbow": elbow_path,
        "silhouette": sil_path,
        "pca": pca_png,
        "centroids": centroids_csv,
        "clustered": clustered_csv,
        "summary": summary_md
    }
