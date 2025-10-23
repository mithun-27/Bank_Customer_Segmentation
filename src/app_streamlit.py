import os
import pandas as pd
import plotly.express as px
import streamlit as st

from config import PATHS

st.set_page_config(page_title="Bank Customer Segmentation", layout="wide")
st.title("Bank Customer Segmentation — Interactive Explorer")

clustered_csv = os.path.join(PATHS.reports_dir, "data_clustered.csv")
centroids_csv = os.path.join(PATHS.reports_dir, "centroids.csv")

if not os.path.exists(clustered_csv):
    st.warning("Please run `python run_pipeline.py` first to generate outputs.")
    st.stop()

df = pd.read_csv(clustered_csv)
st.subheader("Clustered Data")
st.dataframe(df.head(50))

# PCA scatter
st.subheader("PCA Scatter")
cluster_col = "Cluster"
fig = px.scatter(
    df, x="PCA1", y="PCA2", color=cluster_col,
    hover_data=[c for c in df.columns if c not in ["PCA1","PCA2"]]
)
st.plotly_chart(fig, use_container_width=True)

# Centroids
if os.path.exists(centroids_csv):
    st.subheader("Cluster Centroids (Original Scale)")
    cents = pd.read_csv(centroids_csv)
    st.dataframe(cents)
