#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation
"""
import logging
import scipy.signal.windows as windows
import scipy.signal
scipy.signal.hann = windows.hann

from soundsig.signal import lowpass_filter, bandpass_filter, resample_signal
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import torch
import torchaudio
from joblib import Parallel, delayed
from typing import Union
from utils import create_tsv, dump_features, get_km_label, learn_kmeans


def _init_logger(debug=False):
    message_fmt = "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument("--dataset", default="short_zebra_finch", type=str, choices=["librispeech", "librilight","short_zebra_finch"])
    parser.add_argument(
        "--root-dir",
        default="/Users/jonathanwang/Desktop/vocalizations_lab/datasets/Zebra_Finch_Dataset/dataset",
        type=Path,
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num-rank", default=5, type=int)
    #TODO: What are these arguments?
    parser.add_argument("--feat-type", default="mfcc", choices=["mfcc", "hubert"], type=str)
    parser.add_argument(
        "--layer-index",
        default=6,
        type=int,
        help="The layer index in HuBERT model for feature extraction. (``1`` means the first layer output)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument("--use-gpu", default=False, type=bool)
    parser.add_argument(
        "--exp-dir",
        default="/Users/jonathanwang/Desktop/vocalizations_lab/HubertRes",
        type=Path,
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num-cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="The percent of data for KMeans clustering. If negative, use all data. (Default: -1)",
    )
    parser.add_argument(
        "--valid",
        default=False,
        type=bool,
        help="Whether to create a validation set. (Default: False)",
    )
    args = parser.parse_args()
    return args


def resample_and_save_audio(input_path: Path, output_path: Path, orig_freq: int, new_freq: int,
                            low_freq=100, high_freq=8000, filter_order=5, rescale=False, chunk_size=10):
    waveform, sr = torchaudio.load(input_path)
    assert sr == orig_freq, "Sample rate mismatch"

    chunk_samples = orig_freq * chunk_size
    num_chunks = (waveform.size(1) + chunk_samples - 1) // chunk_samples  # Ceiling division
    if waveform.shape[0] > 1:
        # Take the mean across the channels (shape: (n_channels, n_time) -> (n_time))
        waveform = waveform.mean(dim=0)
        waveform = waveform.unsqueeze(0)  # Now shape is [1, samples]

    resampled_waveform = []

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, waveform.size(1))
        chunk = waveform[:, start:end]
        resampled_chunk = bandpass_filter(chunk, sr, low_freq, high_freq, filter_order, rescale).squeeze(0)
        t_rs, filtered_chunk = resample_signal(resampled_chunk, sr, new_freq)
        resampled_waveform.append(torch.tensor(filtered_chunk))

    # Concatenate all chunks along the time dimension
    resampled_waveform = torch.cat(resampled_waveform).unsqueeze(0)

    # Save the resampled and filtered waveform
    torchaudio.save(output_path, resampled_waveform, new_freq)


def preprocess_and_save_all(tsv_file: Union[str, Path], output_dir: Union[str, Path], orig_freq: int, new_freq: int):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]

    def preprocess_and_save(line):
        path, nsample = line.split("\t")
        input_path = Path(root) / path
        output_path = output_dir / Path(path).name

        resample_and_save_audio(input_path, output_path, orig_freq, new_freq)

        # Update the path to the new preprocessed file
        return f"{output_path.relative_to(output_dir.parent)}\t{nsample}"

    new_lines = Parallel(n_jobs=-1)(delayed(preprocess_and_save)(line) for line in lines)

    return new_lines


def main(args):
    _init_logger(args.debug)

    if not args.exp_dir.exists():
        args.exp_dir.mkdir()
    if args.feat_type == "mfcc":
        data_dir = args.exp_dir / "data" / "mfcc"
    else:
        data_dir = args.exp_dir / "data" / f"{args.feat_type}_{args.layer_index}"
    data_dir.mkdir(parents=True, exist_ok=True)

    split_sets = ["train"]
    if args.valid:
        split_sets.append("valid")

    tsv_dir = data_dir / "tsv"
    feat_dir = data_dir / "feat"
    km_dir = data_dir / "km_model"
    label_dir = data_dir / "label"

    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create file lists for training and validation (optional)
    create_tsv(args.root_dir, tsv_dir, args.dataset, extension="wav")

    # Preprocess and save audio files
    preprocessed_audio_dir = data_dir / "preprocessed_audio"
    for split in split_sets:
        new_lines = preprocess_and_save_all(
            tsv_dir / f"{args.dataset}_{split}.tsv",
            preprocessed_audio_dir,
            orig_freq=44100,
            new_freq=16000
        )

        # Delete old TSV file
        old_tsv_file = tsv_dir / f"{args.dataset}_{split}.tsv"
        old_tsv_file.unlink()

        # Write new TSV file with preprocessed audio paths
        with open(old_tsv_file, "w") as f:
            f.write(f"{data_dir}\n")
            for new_line in new_lines:
                f.write(f"{new_line}\n")

    # Extract features for KMeans clustering
    if not feat_dir.exists():
        feat_dir.mkdir()

    for split in split_sets:
        Parallel(n_jobs=-1)(delayed(dump_features)(
            tsv_dir / f"{args.dataset}_{split}.tsv",
            feat_dir,
            split,
            rank,
            args.num_rank,
            device,
            args.feat_type,
            args.layer_index,
            args.checkpoint_path,
            16_000,
        ) for rank in range(1, args.num_rank + 1))

    # Fit KMeans clustering model
    learn_kmeans(
        feat_dir,
        "train",
        args.num_rank,
        km_dir,
        args.num_cluster,
        args.percent,
    )

    # Predict labels for MFCC or HuBERT features
    for split in split_sets:
        get_km_label(
            feat_dir,
            km_dir,
            label_dir,
            split,
            args.num_rank,
            device,
        )


if __name__ == "__main__":
    main(_parse_args())
