import pandas as pd
import matplotlib.pyplot as plt
import random

# Load your data
df = pd.read_csv("DBSCAN_graph_paper.csv")

# Filter only high-risk rows
df_high = df[df["risk"] == "high"]

# Get unique clusters
unique_clusters = df_high["cluster"].unique()

# Generate random colors for each cluster
def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

cluster_colors = {cluster: random_color() for cluster in unique_clusters}

# Map colors to the dataframe
df_high["color"] = df_high["cluster"].map(cluster_colors)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(
    df_high["timestamp"],
    df_high["cluster"],
    s=100,
    c=df_high["color"],
    edgecolors="black"
)

plt.xlabel("Timestamp")
plt.ylabel("Cluster ID")
plt.title("High-Risk Events by Cluster (Random Colors)")
plt.grid(False)
plt.tight_layout()
plt.show()