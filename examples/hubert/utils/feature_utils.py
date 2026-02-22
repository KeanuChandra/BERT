#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from soundsig.sound import spectrogram
from torch.nn import Module

from .common_utils import _get_feat_lens_paths

_LG = logging.getLogger(__name__)
_DEFAULT_DEVICE = torch.device("cpu")
output_specs = 0


def _save_first_chunk_spectrogram(chunk, log_spec, audio_path, freq, start_idx, sample_rate, output_dir=None):
    """Save spectrogram plot for the first chunk for verification purposes."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use('Agg')  # Non-interactive backend
        from pathlib import Path

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot waveform
        time_axis = np.linspace(start_idx/sample_rate, (start_idx + len(chunk))/sample_rate, len(chunk))
        ax1.plot(time_axis, chunk, linewidth=0.5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'First Chunk Waveform - {Path(audio_path).name}')
        ax1.grid(True, alpha=0.3)

        # Plot spectrogram
        im = ax2.imshow(
            log_spec.numpy(),
            aspect='auto',
            origin='lower',
            extent=[0, log_spec.shape[1], 0, freq[-1]],
            cmap='viridis'
        )

        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title(f'First Chunk: 1ms Resolution Spectrogram\n{log_spec.shape[0]} freq bins × {log_spec.shape[1]} time frames')

        plt.colorbar(im, ax=ax2, label='Log Power')
        plt.tight_layout()

        # Save to output_dir/debug/ if provided, otherwise /tmp/
        if output_dir:
            debug_dir = Path(output_dir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            output_path = debug_dir / f"first_chunk_spectrogram_{Path(audio_path).stem}.png"
        else:
            output_path = f"/tmp/first_chunk_spectrogram_{Path(audio_path).stem}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        _LG.info(f"First chunk spectrogram saved to: {output_path}")

    except Exception as e:
        _LG.warning(f"Could not save first chunk spectrogram: {e}")

def get_shard_range(num_lines: int, num_rank: int, rank: int) -> Tuple[int, int]:
    r"""Get the range of indices for the current rank in multi-processing.
    Args:
        num_lines (int): The number of lines to process.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        rank (int): The rank in the multi-processing.

    Returns:
        (int, int):
        int: The start index for the current rank.
        int: The end index for the current rank.
    """
    assert 1 <= rank <= num_rank, f"invalid rank/num_rank {rank}/{num_rank}"
    assert num_lines > 0, f"Found {num_lines} files, make sure you specify the correct root directory"
    start = round(num_lines / num_rank * (rank - 1))
    end = round(num_lines / num_rank * rank)
    _LG.info(f"rank {rank} of {num_rank}, process {end-start} " f"({start}-{end}) out of {num_lines}")
    return start, end


def extract_feature_spectrogram(
    path: str,
    device: torch.device,
    sample_rate: int,
    kernel_size_ms: int = 25,
    stride_ms: int = 20,
    debug: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tensor:
    r"""Extract 1ms resolution spectrogram features using soundsig.sound.spectrogram (Gaussian STFT)
    optimized for birdsong analysis, for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform.to(device).squeeze(0).cpu().numpy()

    # Process in 5-second chunks to handle very long audio files
    segment_length = 5  # seconds
    segment_samples = sample_rate * segment_length

    spec_sample_rate = 1000  # 1ms time resolution = 1000Hz sampling rate
    freq_spacing = 50  # Frequency resolution in Hz (good for birdsong analysis)

    # Overlap between chunks to ensure seamless windowing at boundaries
    # Need at least kernel_size_ms worth of overlap (in samples)
    overlap_ms = kernel_size_ms  # 25ms overlap
    overlap_samples = int(overlap_ms * sample_rate / 1000)  # 400 samples at 16kHz

    all_features = []
    total_samples = len(waveform)

    # Track position in the global timeline for proper windowing
    global_frame_offset = 0
    chunk_idx = 0
    chunk_start = 0

    while chunk_start < total_samples:
        # Extract chunk with overlap from previous chunk's end
        chunk_end = min(chunk_start + segment_samples, total_samples)
        chunk = waveform[chunk_start:chunk_end]

        # Generate spectrogram for this chunk
        t, freq, timefreq, rms = spectrogram(
            chunk,
            sample_rate,
            spec_sample_rate=spec_sample_rate,
            freq_spacing=freq_spacing,
            cmplx=False  # Return magnitude, not complex
        )

        # Convert to power spectrogram and log scale
        spec_data = timefreq ** 2  # Power spectrogram
        log_spec = torch.tensor(spec_data + 1e-8).log10()

        # Save spectrogram plot for first chunk only (for debugging/verification)
        if chunk_idx == 0 and debug:
            _save_first_chunk_spectrogram(chunk, log_spec, path, freq, chunk_start, sample_rate, output_dir)

        # Extract 25ms windows from this chunk's spectrogram
        time_frames = log_spec.shape[1]
        frames_per_window = kernel_size_ms  # 25ms = 25 frames
        frames_per_hop = stride_ms          # 20ms = 20 frames

        # For chunks after the first, start at the correct stride-aligned position
        # to avoid duplicates while not missing any windows
        if chunk_idx == 0:
            start_frame = 0
        else:
            # Find the first stride-aligned global window position that wasn't
            # covered by the previous chunk.
            # Global windows are at 0, stride_ms, 2*stride_ms, ... ms.
            # This chunk starts at chunk_start_ms; we need the first global
            # window >= chunk_start_ms + overlap_ms, rounded up to the next stride.
            # Formula: start_frame = stride_ms - (chunk_start_ms % stride_ms),
            #   using stride_ms when chunk_start_ms is already stride-aligned.
            # Example (stride=20, overlap=25, step=4975ms):
            #   Chunk 1 starts at 4975ms: r=15 -> start_frame=5  (global 4980ms)
            #   Chunk 2 starts at 9950ms: r=10 -> start_frame=10 (global 9960ms)
            #   Chunk 3 starts at 14925ms: r=5 -> start_frame=15 (global 14940ms)
            #   Chunk 4 starts at 19900ms: r=0 -> start_frame=20 (global 19920ms)
            chunk_start_ms = (chunk_start * 1000) // sample_rate
            r = chunk_start_ms % stride_ms
            start_frame = stride_ms - r if r != 0 else stride_ms

        # Extract overlapping windows
        chunk_features = []
        for frame_pos in range(start_frame, time_frames - frames_per_window + 1, frames_per_hop):
            end_frame = frame_pos + frames_per_window

            # Extract 25ms window from this chunk's spectrogram
            window = log_spec[:, frame_pos:end_frame]  # Shape: (freq_bins, 25)

            # Flatten to vector for KMeans
            feature_vector = window.flatten()  # Shape: (freq_bins * 25,)
            chunk_features.append(feature_vector)

        all_features.extend(chunk_features)

        # Move to next chunk, stepping back by overlap to maintain continuity
        chunk_start = chunk_end - overlap_samples
        chunk_idx += 1

        # Prevent infinite loop if we're at the end
        if chunk_end >= total_samples:
            break

    if len(all_features) == 0:
        # Handle edge case of very short audio
        default_freq_bins = 160  # Default frequency bins for 8kHz with 50Hz spacing
        return torch.zeros(1, default_freq_bins * kernel_size_ms)

    # Stack all feature vectors
    feature_matrix = torch.stack(all_features)  # Shape: (num_windows, feature_dim)

    return feature_matrix


def extract_feature_hubert(
    path: str,
    device: torch.device,
    sample_rate: int,
    model: Module,
    layer_index: int,
) -> Tensor:
    r"""Extract HuBERT features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.
        model (Module): The loaded ``HuBERTPretrainModel`` model.
        layer_index (int): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output).

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform.to(device)
    with torch.inference_mode():
        feat = model.wav2vec2.extract_features(waveform, num_layers=layer_index)[0][-1][0]  # (time, feat_dim)
    return feat


def _load_state(model: Module, checkpoint_path: Path, device=_DEFAULT_DEVICE) -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model


def dump_features(
    tsv_file: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "spectrogram",
    layer_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    sample_rate: int = 16_000,
    kernel_size_ms: int = 25,
    stride_ms: int = 20,
    debug: bool = False,
) -> None:
    r"""Dump the feature tensors given a ``.tsv`` file list. The feature and lengths tensors
        will be stored under ``out_dir`` directory.
    Args:
        tsv_file (str or Path): The path of the tsv file.
        out_dir (str or Path): The directory to store the feature tensors.
        split (str): The split of data. Options: [``train``, ``valid``].
        rank (int): The rank in the multi-processing.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        feature_type (str, optional): The type of the desired feature. Options: [``spectrogram``, ``hubert``].
            (Default: ``spectrogram``)
        layer_index (int or None, optional): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output). Only active when ``feature_type``
            is set to ``hubert``. (Default: ``None``)
        checkpoint_path(Path or None, optional): The checkpoint path of ``torchaudio.models.HuBERTPretrainModel``.
            Only active when ``feature_type`` is set to ``hubert``. (Default: ``None``)
        sample_rate (int, optional): The sample rate of the audio. (Default: ``16000``)

    Returns:
        None
    """
    if feature_type not in ["spectrogram", "hubert"]:
        raise ValueError(f"Expected feature type to be 'spectrogram' or 'hubert'. Found {feature_type}.")
    if feature_type == "hubert" and layer_index is None:
        assert ValueError("Please set the layer_index for HuBERT feature.")
    features = []
    lens = []
    out_dir = Path(out_dir)

    feat_path, len_path = _get_feat_lens_paths(out_dir, split, rank, num_rank)

    if feature_type == "hubert":
        from torchaudio.models import hubert_pretrain_base

        model = hubert_pretrain_base()
        model.to(device)
        model = _load_state(model, checkpoint_path, device)

    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]

        num_files = len(lines)
        if num_files == 0:
            # No files to process
            start, end = 0, 0
        elif num_files < num_rank:
            # Fewer files than ranks: only first num_files ranks get work
            # Each rank gets at most 1 file, remaining ranks get nothing
            if rank <= num_files:
                start, end = rank - 1, rank  # rank is 1-indexed
            else:
                start, end = 0, 0  # This rank has nothing to do
                _LG.info(f"Rank {rank} of {num_rank}: no files to process (only {num_files} files)")
        else:
            # Normal case: distribute files evenly across ranks
            start, end = get_shard_range(num_files, num_rank, rank)

        lines = lines[start:end]
        for line in lines:
            path, nsample = line.split("\t")
            path = f"{root}/{path}"
            if feature_type == "spectrogram":
                feature = extract_feature_spectrogram(path, device, sample_rate, kernel_size_ms, stride_ms, debug, out_dir)
            else:
                feature = extract_feature_hubert(path, device, sample_rate, model, layer_index)
            features.append(feature.cpu())
            lens.append(feature.shape[0])
    if len(features) != 0:
        features = torch.cat(features)
    else:
        features = torch.empty(0, 0)
    lens = torch.Tensor(lens)
    torch.save(features, feat_path)
    torch.save(lens, len_path)
    _LG.info(f"Finished dumping features for rank {rank} of {num_rank} successfully")
