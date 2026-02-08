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
    total_duration = len(waveform) / sample_rate

    spec_sample_rate = 1000  # 1ms time resolution = 1000Hz sampling rate
    freq_spacing = 50  # Frequency resolution in Hz (good for birdsong analysis)

    all_features = []
    num_chunks = (len(waveform) + segment_samples - 1) // segment_samples

    for chunk_idx in range(num_chunks):
        # Extract 5-second chunk
        start_idx = chunk_idx * segment_samples
        end_idx = min(start_idx + segment_samples, len(waveform))
        chunk = waveform[start_idx:end_idx]

        # Generate FULL spectrogram for this 5-second chunk
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
            _save_first_chunk_spectrogram(chunk, log_spec, path, freq, start_idx, sample_rate, output_dir)

        # Extract 25ms windows from this chunk's spectrogram
        time_frames = log_spec.shape[1]
        frames_per_window = kernel_size_ms  # 25ms = 25 frames
        frames_per_hop = stride_ms          # 20ms = 20 frames

        # Extract overlapping windows: 0-25ms, 20-45ms, 40-65ms, etc.
        chunk_features = []
        for start_frame in range(0, time_frames - frames_per_window + 1, frames_per_hop):
            end_frame = start_frame + frames_per_window

            # Extract 25ms window from this chunk's spectrogram
            window = log_spec[:, start_frame:end_frame]  # Shape: (freq_bins, 25)

            # Flatten to vector for KMeans
            feature_vector = window.flatten()  # Shape: (freq_bins * 25,)
            chunk_features.append(feature_vector)

        all_features.extend(chunk_features)

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
        start, end = 0, 0
        if len(lines) == 1:
            start, end = 0, 1
        elif len(lines) != 0:
            start, end = get_shard_range(len(lines), num_rank, rank)
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
