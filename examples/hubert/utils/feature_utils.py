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
from torchaudio.transforms import Spectrogram
from torch.nn import Module
import torch.nn.functional as F
# from soundsig.signal import spectrogram, plot_spectrogram

from .common_utils import _get_feat_lens_paths

_LG = logging.getLogger(__name__)
_DEFAULT_DEVICE = torch.device("cpu")
debug = False
output_specs = 0

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


def extract_feature_mfcc(
    path: str,
    device: torch.device,
    sample_rate: int,
) -> Tensor:
    r"""Extract MFCC features for KMeans clustering and pseudo label prediction.
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
    waveform = waveform.to(device)

    # Use 5-second segments for processing, or entire file if shorter
    segment_length = 5  # seconds
    segment_samples = sample_rate * segment_length

    # Handle files shorter than segment_length
    if waveform.shape[1] <= segment_samples:
        # Pad the waveform to segment_samples length and process as one segment
        padding_needed = segment_samples - waveform.shape[1]
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        segments = padded_waveform.unsqueeze(1)  # Shape: (channels, 1, segment_samples)
    else:
        # Split into segments
        segments = waveform.unfold(dimension=1, size=segment_samples, step=segment_samples)

    # Create spectrogram transform with 5ms hop length (80 samples at 16kHz)
    spectrogram_transform = Spectrogram(n_fft=400, hop_length=80, center=False).to(device)

    def f(segment):
        # Apply spectrogram transform
        spec = spectrogram_transform(segment.unsqueeze(0))  # Add batch dim
        spec = spec.squeeze(0)  # Remove batch dim
        spec = torch.log10(spec + 1e-8)  # Log scale with numerical stability
        return spec.cpu().numpy()

    spectrograms = []
    for i in range(segments.shape[1]):
        segment = segments[:, i, :]  # (channel, segment_samples)
        # Handle case where segment might be 1D or 2D
        if segment.dim() > 1:
            segment = segment.squeeze(0)  # Remove channel dimension if present
        spec = f(segment)
        spectrograms.append(torch.from_numpy(spec))

    spectrograms = torch.stack(spectrograms)  # not cat

    # spectrograms = vmap(f)(segments)

    frames_per_25ms = 5  # hop length is 5ms so 5 hop lengths = 25ms
    frames_per_20ms = 4  # hop length is 5ms so 4 hop lengths = 20ms

    # Updated: Use sliding windows for 25ms segments with 20ms step (matches CNN encoder)
    window_size = frames_per_25ms  # 5 frames = 25ms window
    step_size = frames_per_20ms    # 4 frame step = 20ms (matches CNN stride)
    features = []

    # spectrograms shape: (num_segments, freq_bins, time_frames)
    # We want to window over time_frames (dimension 2), not freq_bins (dimension 1)
    for i in range(0, spectrograms.shape[2] - window_size + 1, step_size):
        window = spectrograms[:, :, i:i+window_size]  # (num_segments, freq_bins, window_size)
        features.append(window)

    flattened = [t.reshape(t.shape[0], -1) for t in features]  # each becomes (N, M)
    target_len = flattened[0].shape[1]
    flattened = [F.pad(t, (0, target_len - t.shape[1])) if t.shape[1] < target_len else t for t in flattened]
    stacked = torch.cat(flattened, dim=0)  # final shape: (len(tensor_list) * 1136, M)

    features = stacked

    # if spectrogram.size(1) == 1:
    #     spectrogram = spectrogram.expand(-1, 3, -1)

    # waveform = waveform[0].to(device)
    # mfccs = feature_extractor(waveform)  # (freq, time)
    # deltas = torchaudio.functional.compute_deltas(mfccs)
    # ddeltas = torchaudio.functional.compute_deltas(deltas)
    # concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
    # feat = concat.transpose(0, 1)  # (time, freq)
    # import matplotlib.pyplot as plt
    #
    # spec = spectrogram.cpu().numpy()  # Convert to numpy for plotting
    # plt.figure(figsize=(10, 6))
    #
    # # Plot the first channel (in case of multi-channel audio)
    # plt.imshow(spectrogram[:, :, 0], aspect='auto', origin='lower')
    # plt.colorbar(format='%+2.0f dB')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Frequency (bins)')
    # plt.title('Spectrogram')
    # plt.show()
    return features


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
    feature_type: str = "mfcc",
    layer_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    sample_rate: int = 16_000,
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
        feature_type (str, optional): The type of the desired feature. Options: [``mfcc``, ``hubert``].
            (Default: ``mfcc``)
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
    if feature_type not in ["mfcc", "hubert"]:
        raise ValueError(f"Expected feature type to be 'mfcc' or 'hubert'. Found {feature_type}.")
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
            if feature_type == "mfcc":
                feature = extract_feature_mfcc(path, device, sample_rate)
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
