import pandas as pd


df = pd.read_csv("110504-000_clusters_metadata.csv")

cluster_list = df["cluster"].astype(int).tolist()

plain_text_clusters = " ".join(map(str, cluster_list))

output_filename = "110504-000_clusters.pt"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(plain_text_clusters)

print(f"Successfully converted {len(cluster_list)} cluster values!")
print(f"Saved plain-text data masked as: {output_filename}")