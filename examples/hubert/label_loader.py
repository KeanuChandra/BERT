import pandas as pd
import torch  


df = pd.read_csv("110504-000_clusters_metadata.csv")


cluster_list = df["cluster"].astype(int).tolist()

cluster_tensor = torch.tensor(cluster_list, dtype=torch.long)

output_filename = "110504-000_clusters.pt"
torch.save(cluster_tensor, output_filename)

print(f"Successfully converted {len(cluster_list)} cluster values!")
print(f"Saved a genuine binary PyTorch tensor to: {output_filename}")