import os
import math
import torch
import torchaudio
import torchaudio.transforms as T

def chunk_dataset(audio_path, label_path, output_dir, segment_secs=10, sample_rate=16000, ms_per_frame=20):
    print("Starting dataset splitting pipeline...")
    
    # 1. Setup output directory structure
    audio_out_dir = os.path.join(output_dir, "audio")
    tsv_out_dir = os.path.join(output_dir, "tsv")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(tsv_out_dir, exist_ok=True)
    
    # 2. Load and RESAMPLE Audio
    print(f"-> Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    
    if sr != sample_rate:
        print(f"-> Resampling audio from {sr}Hz to {sample_rate}Hz...")
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    # 3. Load Labels from PyTorch Binary .pt File
    print(f"-> Loading binary labels via PyTorch: {label_path}")
    labels = torch.load(label_path, map_location="cpu")
    labels = labels.long().flatten()

    # 4. Calculate framing parameters
    samples_per_seg = segment_secs * sample_rate
    frames_per_seg = int((segment_secs * 1000) / ms_per_frame)
    
    total_samples = waveform.size(1)
    total_frames = labels.size(0)
    
    # Calculate how many full segments we can make
    num_segments = min(
        math.floor(total_samples / samples_per_seg),
        math.floor(total_frames / frames_per_seg)
    )
    
    print(f"Resampled Samples: {total_samples} | Total Label Frames: {total_frames}")
    print(f"Slicing into {num_segments} chunks of {segment_secs}s each...")
    
    chunked_labels = []
    tsv_lines = []
    
    # 5. Slice simultaneously 
    for i in range(num_segments):
        # Slice Audio
        start_sample = i * samples_per_seg
        end_sample = start_sample + samples_per_seg
        audio_chunk = waveform[:, start_sample:end_sample]
        
        chunk_filename = f"chunk_{i:04d}.wav"
        chunk_filepath = os.path.join(audio_out_dir, chunk_filename)
        torchaudio.save(chunk_filepath, audio_chunk, sample_rate)
        
        # Slice Labels
        start_frame = i * frames_per_seg
        end_frame = start_frame + frames_per_seg
        label_chunk = labels[start_frame:end_frame]
        chunked_labels.append(label_chunk)
        
        tsv_lines.append(f"{chunk_filename}\t{samples_per_seg}\n")
        
    # 6. Save the labels as a .txt (one line per chunk)
    label_txt_path = os.path.join(output_dir, "label_train.txt") 
    with open(label_txt_path, "w", encoding="utf-8") as f:
        for chunk in chunked_labels:
            line_str = " ".join(str(int(x)) for x in chunk)
            f.write(line_str + "\n")
            
    # 7. Write manifest TSV
    tsv_path = os.path.join(tsv_out_dir, "train.tsv")
    with open(tsv_path, "w") as f:
        f.write(f"{os.path.abspath(audio_out_dir)}\n")
        f.writelines(tsv_lines)
        
    print(f"-> Manifest: {tsv_path}")
    print(f"-> Labels: {label_txt_path}")
    print("🎉 Done!")

if __name__ == "__main__":
    INPUT_AUDIO = r"C:\Users\oscar\Documents\BERT\BERT\SpectrogramBasedBERT\examples\hubert\my_mini_dataset\110504-000.wav"
    INPUT_LABELS = r"C:\Users\oscar\Documents\BERT\BERT\SpectrogramBasedBERT\examples\hubert\my_mini_dataset\110504-000_clusters.pt"
    OUTPUT_DATASET_DIR = r"C:\Users\oscar\Documents\BERT\BERT\SpectrogramBasedBERT\examples\hubert\output\my_mini_dataset"
    
    chunk_dataset(
        audio_path=INPUT_AUDIO,
        label_path=INPUT_LABELS,
        output_dir=OUTPUT_DATASET_DIR,
        segment_secs=10, 
        sample_rate=16000
    )