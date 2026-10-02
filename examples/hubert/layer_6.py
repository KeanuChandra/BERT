import os
import sys
import csv
import torch
import torchaudio
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
    print(f"CUDA context initialized successfully on: {torch.cuda.get_device_name(0)}")
# Ensure local modules can be imported
sys.path.append("/global/scratch/users/keanumchandra/BERT/examples/hubert")
from lightning_modules import HuBERTPreTrainModule

def load_model_from_checkpoint(ckpt_path, device):
    """Directly loads checkpoint state dict onto an instantiated HuBERT module."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})

    kwargs = {
        "model_name": "hubert_pretrain_base",
        "feature_grad_mult": 0.1,
        "num_classes": 100,  # Must be an integer for torchaudio
        "dataset": "librispeech",
        "dataset_path": "/global/scratch/users/keanumchandra/split_audio_2",
        "feature_type": "mfcc",
        "seconds_per_batch": 60.0,
        "learning_rate": 5e-4,
        "betas": (0.9, 0.98),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "clip_norm": 10.0,
        "warmup_updates": 32000,
        "max_updates": 250000,
    }

    # Override defaults with saved hparams
    kwargs.update(hparams)

    # Ensure num_classes is an int, even if hparams stores it as [100]
    if isinstance(kwargs.get("num_classes"), list):
        kwargs["num_classes"] = kwargs["num_classes"][0]

    print(f"Instantiating module with model_name='{kwargs['model_name']}' and num_classes={kwargs['num_classes']}...")
    model = HuBERTPreTrainModule(**kwargs)

    # Load state dict safely
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned_state_dict[key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def extract_l6_features(model, tsv_file, device):
    """Extracts Layer 6 frame features for all audio files in a TSV manifest."""
    if not os.path.exists(tsv_file):
        print(f"Skipping missing TSV file: {tsv_file}")
        return [], []

    with open(tsv_file, "r") as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]

    root_dir = lines[0][0]
    audio_entries = lines[1:]

    file_features = []
    file_paths = []
    print(f"Extracting Layer 6 features from {len(audio_entries)} files in {os.path.basename(tsv_file)}...")

    with torch.no_grad():
        for entry in audio_entries:
            rel_path = entry[0]
            wav_path = os.path.join(root_dir, rel_path)

            if not os.path.exists(wav_path):
                continue

            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            waveform = waveform.to(device)
            audio_len = torch.tensor([waveform.shape[1]], device=device)

            # Feature extractor (CNN)
            x, out_len = model.model.wav2vec2.feature_extractor(waveform, audio_len)
            x, attention_mask = model.model.wav2vec2.encoder._preprocess(x, out_len)

            # Pass through Transformer up to Layer 6
            layers = model.model.wav2vec2.encoder.transformer.layers[:6]
            for layer in layers:
                x, _ = layer(x, attention_mask=attention_mask)

            feats = x.squeeze(0).cpu().numpy()
            file_features.append(feats)
            file_paths.append(rel_path)

    return file_features, file_paths


def main():
    CKPT_PATH = "/global/scratch/users/keanumchandra/BERT/examples/hubert/exp/checkpoints_librispeech_hubert_pretrain_base/epoch=73-step=5180.ckpt"
    DATASET_DIR = "/global/scratch/users/keanumchandra/split_audio_2"
    NUM_CLUSTERS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    # 1. Load Model
    model = load_model_from_checkpoint(CKPT_PATH, DEVICE)

    # 2. Extract Features
    train_tsv = os.path.join(DATASET_DIR, "train.tsv")
    valid_tsv = os.path.join(DATASET_DIR, "valid.tsv")

    train_feats, train_paths = extract_l6_features(model, train_tsv, DEVICE)
    valid_feats, valid_paths = extract_l6_features(model, valid_tsv, DEVICE)

    if not train_feats:
        raise ValueError(f"No features extracted from {train_tsv}")

    # 3. Fit K-Means
    print(f"\nFitting MiniBatchKMeans with {NUM_CLUSTERS} clusters...")
    all_train_frames = np.concatenate(train_feats, axis=0)
    print(f"Total frame vectors for training K-Means: {all_train_frames.shape}")

    kmeans = MiniBatchKMeans(
        n_clusters=NUM_CLUSTERS,
        batch_size=10000,
        random_state=42,
        n_init="auto",
    )
    kmeans.fit(all_train_frames)

    # Save K-Means model
    model_output_path = "/global/scratch/users/keanumchandra/km_l6_500.model"
    joblib.dump(kmeans, model_output_path)
    print(f"Saved K-Means model to: {model_output_path}")

    # 4. Export Cluster Centroids to TSV
    centroids_tsv_path = os.path.join(DATASET_DIR, "cluster_centroids.tsv")
    print(f"Exporting cluster centroids to: {centroids_tsv_path}")
    with open(centroids_tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["cluster_id"] + [f"dim_{i}" for i in range(kmeans.cluster_centers_.shape[1])]
        writer.writerow(header)
        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            writer.writerow([cluster_id] + list(center))

    # 5. Export Cluster Assignments to TSV & .km files
    splits = [("train", train_feats, train_paths), ("valid", valid_feats, valid_paths)]

    for split_name, feats_list, paths_list in splits:
        if not feats_list:
            continue

        tsv_out_path = os.path.join(DATASET_DIR, f"{split_name}_clusters.tsv")
        km_out_path = os.path.join(DATASET_DIR, f"{split_name}.km")

        print(f"Writing TSV cluster manifest to: {tsv_out_path}")
        print(f"Writing space-delimited .km to: {km_out_path}")

        with open(tsv_out_path, "w", newline="") as tsv_f, open(km_out_path, "w") as km_f:
            tsv_writer = csv.writer(tsv_f, delimiter="\t")
            # TSV Header: audio_path, total_frames, cluster_sequence (tab-delimited)
            tsv_writer.writerow(["audio_path", "num_frames", "cluster_ids"])

            for rel_path, file_feat in zip(paths_list, feats_list):
                cluster_ids = kmeans.predict(file_feat)
                cluster_str_tab = "\t".join(map(str, cluster_ids))
                cluster_str_space = " ".join(map(str, cluster_ids))

                # Write TSV entry
                tsv_writer.writerow([rel_path, len(cluster_ids), cluster_str_tab])

                # Write standard .km line
                km_f.write(cluster_str_space + "\n")

    # 6. Create dict.km.txt
    dict_path = os.path.join(DATASET_DIR, "dict.km.txt")
    print(f"Writing unit dictionary to: {dict_path}")
    with open(dict_path, "w") as f:
        for i in range(NUM_CLUSTERS):
            f.write(f"{i} 1\n")

    print("\nExtraction & TSV generation complete!")


if __name__ == "__main__":
    main()
