import torch
import os
from pathlib import Path

# Load your long label tensor (replace with actual path)
with open("/global/scratch/users/jonathanswang/temp_files/data/mfcc/label/label_train.pt") as f:
    lines = [line.strip() for line in f]
    long_labels = torch.tensor([int(x) for line in lines for x in line.split()])  # flatten all chunks
print(long_labels.shape)

# Sanity check
# assert long_labels.numel() == 566 * 500, "Label tensor length mismatch"

# Split and save
num_chunks = 566
chunk_size = 500

output_path = Path("/global/scratch/users/jonathanswang/temp_files/data/mfcc/label/label_train_new.pt")

# Ensure parent directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Write chunks as space-separated lines
with open(output_path, "w") as f:
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = long_labels[start:end]  # shape: (500,)
        line = " ".join(str(x.item()) for x in chunk)
        f.write(line + "\n")


