
import os
import math
import pandas as pd

def generate_labels_from_csv(csv_path, output_dir, segment_secs=10, ms_per_frame=20):
    print("Starting label formatting and chunk alignment from CSV...")

    # 1. Ensure the specific destination folder expected by your loader exists
    target_dir = os.path.join(output_dir, "label")
    os.makedirs(target_dir, exist_ok=True)

    # 2. Read the CSV data
    print(f"-> Reading CSV metadata: {csv_path}")
    df = pd.read_csv(csv_path)

    # 3. Extract clusters as integers
    cluster_list = df["cluster"].astype(int).tolist()
    total_frames = len(cluster_list)

    # 4. Calculate identical framing parameters from your splitting logic
    frames_per_seg = int((segment_secs * 1000) / ms_per_frame)  # (10 * 1000) / 20 = 500 frames
    num_segments = math.floor(total_frames / frames_per_seg)

    print(f"Total CSV Frames: {total_frames} | Target Frames Per Chunk: {frames_per_seg}")
    print(f"Aligning into {num_segments} consecutive audio-matched lines...")

    # 5. Group the clusters line-by-line (one line per 10-second chunk)
    output_filepath = os.path.join(target_dir, "label_train.pt")

    with open(output_filepath, "w", encoding="utf-8") as f:
        for i in range(num_segments):
            start_frame = i * frames_per_seg
            end_frame = start_frame + frames_per_seg
            chunk_frames = cluster_list[start_frame:end_frame]

            # Recreate space-separated row sequence for this specific chunk
            line_str = " ".join(map(str, chunk_frames))
            f.write(line_str + "\n")

    print(f"-> Successfully created text-based label file at: {output_filepath}")
    print(f"-> Total rows written: {num_segments} (Each containing {frames_per_seg} values)")
    print("Done! Your label rows are now perfectly aligned with your TSV manifest rows.")

if __name__ == "__main__":
    CSV_PATH = "110504-000_clusters_metadata.csv"
    OUTPUT_DATASET_DIR = "/global/scratch/users/keanumchandra/split_audio"

    generate_labels_from_csv(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DATASET_DIR,
        segment_secs=10,
        ms_per_frame=20
    )
