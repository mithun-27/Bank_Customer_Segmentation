import os
import matplotlib.pyplot as plt
import pandas as pd
from .config import PATHS

def ensure_reports_dir():
    os.makedirs(PATHS.reports_dir, exist_ok=True)

def plot_elbow(k_values, sse):
    ensure_reports_dir()
    plt.figure(figsize=(8,4))
    plt.plot(k_values, sse, marker='o')
    plt.title("Elbow Method (SSE)")
    plt.xlabel("K")
    plt.ylabel("SSE (Inertia)")
    plt.grid(True)
    out = os.path.join(PATHS.reports_dir, "elbow_sse.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_silhouette(k_values, sil_scores):
    ensure_reports_dir()
    plt.figure(figsize=(8,4))
    plt.plot(k_values, sil_scores, marker='o')
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    out = os.path.join(PATHS.reports_dir, "silhouette_scores.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_pca_scatter(df_with_pca_and_clusters):
    ensure_reports_dir()
    plt.figure(figsize=(8,6))
    for c in sorted(df_with_pca_and_clusters["Cluster"].unique()):
        part = df_with_pca_and_clusters[df_with_pca_and_clusters["Cluster"] == c]
        plt.scatter(part["PCA1"], part["PCA2"], alpha=0.65, s=24, label=f"Cluster {c}")
    plt.title("Customer Segments — PCA 2D")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.grid(True)
    out = os.path.join(PATHS.reports_dir, "clusters_pca.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def save_summary(best_k, k_values, sse, sil_scores, centroids_csv, clustered_csv):
    ensure_reports_dir()
    md = [
        "# K-Means Segmentation — Summary",
        "",
        f"**Selected K:** {best_k}",
        "",
        "## Files",
        f"- `elbow_sse.png` — Elbow chart",
        f"- `silhouette_scores.png` — Silhouette chart",
        f"- `clusters_pca.png` — PCA 2D scatter",
        f"- `k_selection.csv` — SSE & Silhouette for each K",
        f"- `centroids.csv` — cluster centroids (original scale)",
        f"- `data_clustered.csv` — data with `Cluster`, `PCA1`, `PCA2`",
        "",
        "## Notes",
        "- Clusters are formed on scaled features.",
        "- Centroids are inverse-transformed for interpretation."
    ]
    out = os.path.join(PATHS.reports_dir, "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    # Also save k table
    pd.DataFrame({"K": k_values, "SSE": sse, "Silhouette": sil_scores}).to_csv(
        os.path.join(PATHS.reports_dir, "k_selection.csv"), index=False
    )
    return out
