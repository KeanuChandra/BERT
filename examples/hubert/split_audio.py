import torchaudio
import torch
import os

import logging
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter
from typing import Tuple


class _Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    # To use ArgumentDefaultsHelpFormatter as the formatter_class and
    # RawDescriptionHelpFormatter to add custom formatting to description or epilog.
    # Check: https://stackoverflow.com/a/18462760
    pass


def segment_waveform(wav_path, segment_seconds=10, output_dir="segments"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(wav_path)
    waveform, sr = torchaudio.load(wav_path)  # shape: (1, num_samples)
    total_samples = waveform.shape[1]
    segment_len = int(segment_seconds * sr)

    print(segment_len)

    segments = []
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    for i, start in enumerate(range(0, total_samples, segment_len)):
        end = start + segment_len
        segment = waveform[:, start:end]
        
        print(segment.shape)

        # Pad if it's the last segment and shorter than expected
        if segment.shape[1] < segment_len:
            pad_amount = segment_len - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, pad_amount))

        segment_path = os.path.join(output_dir, f"{base_name}_{i}.wav")
        torchaudio.save(segment_path, segment, sr)
        print(segment_path)
        segments.append((segment_path, segment.shape[1]))

    return segments  # list of (path, num_samples)


def update_tsv(segments, tsv_path, root):
    with open(tsv_path, "w") as f:
        print(tsv_path)
        for path, length in segments:
            f.write(f"{os.path.relpath(path, start=root)}\t{length}\n")


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        type=str,
        required=True,
        help="train or validation step.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ZF_test_pipeline", "mfcc"],
        type=str,
        required=True,
        help="dataset to use.",
    )
    return parser.parse_args()


def cli_main():
    args = _parse_args()
    print(args)
    segments = segment_waveform(f"{args.dataset_path}/preprocessed_audio/{args.dataset}_{args.split}.wav", segment_seconds=10, output_dir="segmented_wavs")
    update_tsv(segments, f"{args.dataset_path}/tsv/{args.dataset}_{args.split}.tsv", args.dataset_path)


if __name__ == "__main__":
    cli_main()

