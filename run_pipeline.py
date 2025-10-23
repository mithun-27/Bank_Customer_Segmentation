"""
Run the full K-Means segmentation pipeline end-to-end.
Steps:
- Reads the first CSV found in ./data
- Auto-selects numeric features (drops ID-like)
- Scales, searches K (2..10), picks best K via silhouette
- Fits KMeans, saves centroids (inverse-transformed)
- Runs PCA & saves scatter
- Writes summary & CSV outputs into ./reports
"""

from src.kmeans_segmentation import run_pipeline

if __name__ == "__main__":
    results = run_pipeline()
    print("Done!")
    print("Best K:", results["best_k"])
    print("Files written:")
    for k,v in results.items():
        if k != "best_k":
            print(f" - {k}: {v}")
