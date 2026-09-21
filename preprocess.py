import os
import librosa
import soundfile as sf
import numpy as np

DATASETS = [
    ("raw_voices", os.path.join("dataset", "real")),
    ("raw_fake_voices", os.path.join("dataset", "fake"))
]

TARGET_SR = 22050       # 22.05 kHz standard
CHUNK_DURATION = 3.0    # Fixed 3-second window for CNN
CHUNK_SAMPLES = int(CHUNK_DURATION * TARGET_SR)

valid_exts = ('.mp3', '.mp4', '.wav', '.m4a', '.mpeg', '.ogg')
total_slices_overall = 0

for input_folder, output_folder in DATASETS:
    print(f"\n========================================")
    print(f"[*] Scanning: {input_folder} -> {output_folder}")
    print(f"========================================")

    if not os.path.exists(input_folder):
        print(f"[!] Skipping '{input_folder}' (Folder does not exist).")
        continue

    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    
    if not files:
        print(f"[!] No audio files found in '{input_folder}'. Skipping.")
        continue

    folder_slices = 0

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # Load and resample to mono 22.05kHz
            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            total_samples = len(audio)
            
            # CASE 1: Audio is shorter than 3 seconds -> Pad with silence to 3.0s
            if total_samples < CHUNK_SAMPLES:
                padding = CHUNK_SAMPLES - total_samples
                padded_audio = np.pad(audio, (0, padding), mode='constant')
                
                out_name = f"{base_name}_chunk00.wav"
                out_path = os.path.join(output_folder, out_name)
                sf.write(out_path, padded_audio, TARGET_SR)
                
                folder_slices += 1
                total_slices_overall += 1
                print(f"[+] Padded short clip: '{filename}' ({total_samples/TARGET_SR:.2f}s -> 3.00s) -> {out_name}")
                continue

            # CASE 2: Audio is 3 seconds or longer -> Slice into 3.0s chunks
            num_chunks = total_samples // CHUNK_SAMPLES
            for i in range(num_chunks):
                start = i * CHUNK_SAMPLES
                end = start + CHUNK_SAMPLES
                chunk = audio[start:end]
                
                out_name = f"{base_name}_chunk{i:02d}.wav"
                out_path = os.path.join(output_folder, out_name)
                sf.write(out_path, chunk, TARGET_SR)
                folder_slices += 1
                total_slices_overall += 1

            # Check if there is a leftover remainder longer than 1 second
            remainder_samples = total_samples % CHUNK_SAMPLES
            if remainder_samples >= int(1.0 * TARGET_SR):
                remainder_chunk = audio[-remainder_samples:]
                padded_remainder = np.pad(remainder_chunk, (0, CHUNK_SAMPLES - remainder_samples), mode='constant')
                out_name = f"{base_name}_chunk{num_chunks:02d}.wav"
                out_path = os.path.join(output_folder, out_name)
                sf.write(out_path, padded_remainder, TARGET_SR)
                folder_slices += 1
                total_slices_overall += 1
                num_chunks += 1
                
            print(f"[+] Processed: '{filename}' -> Created {num_chunks} slices")

        except Exception as e:
            print(f"[!] Could not process '{filename}': {e}")
            
    print(f"[*] Finished '{input_folder}'. Total slices in folder: {folder_slices}")

print(f"\n[DONE] Preprocessing complete. Total slices processed: {total_slices_overall}")