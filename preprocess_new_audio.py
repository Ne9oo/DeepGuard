import os
import librosa
import soundfile as sf

INPUT_FOLDER = "path_to_your_raw_voices_folder"  # Folder containing your 1-min files
OUTPUT_FOLDER = "dataset/real"                    # Target training directory
TARGET_SR = 22050                                # Standard audio sample rate
CHUNK_DURATION = 3.0                             # Clip length in seconds

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
chunk_samples = int(CHUNK_DURATION * TARGET_SR)

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.mp3', '.mp4', '.wav', '.m4a', '.mpeg')):
        file_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # Load and resample to mono 22.05kHz
            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            total_samples = len(audio)
            
            # Slice audio into uniform 3-second segments
            num_chunks = total_samples // chunk_samples
            for i in range(num_chunks):
                start = i * chunk_samples
                end = start + chunk_samples
                chunk = audio[start:end]
                
                out_name = f"{base_name}_chunk{i:02d}.wav"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                sf.write(out_path, chunk, TARGET_SR)
                
            print(f"[+] Processed: {filename} -> {num_chunks} slices created.")
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")