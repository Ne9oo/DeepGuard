import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
# The exact path to the extracted ASVspoof LA dataset
ASVSPOOF_EXTRACTED_DIR = r"C:\Users\Afiq\Downloads\LA\LA"

# Paths to the specific ASVspoof folders
TRAIN_AUDIO_DIR = os.path.join(ASVSPOOF_EXTRACTED_DIR, "ASVspoof2019_LA_train", "flac")
PROTOCOL_FILE = os.path.join(ASVSPOOF_EXTRACTED_DIR, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

# Where to put the sorted files for DeepGuard
DEEPGUARD_DATASET_DIR = "dataset"
REAL_DIR = os.path.join(DEEPGUARD_DATASET_DIR, "real")
FAKE_DIR = os.path.join(DEEPGUARD_DATASET_DIR, "fake")

# FYP Limit: We'll copy 2,000 of each so your laptop doesn't take 3 days to train.
# You can increase this later if you want a more accurate model.
LIMIT_PER_CLASS = 2000 

def setup_directories():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)
    print(f"[*] Created directories: {REAL_DIR} and {FAKE_DIR}")

def sort_asvspoof():
    if not os.path.exists(PROTOCOL_FILE):
        print(f"[-] ERROR: Cannot find protocol file at {PROTOCOL_FILE}")
        return
    if not os.path.exists(TRAIN_AUDIO_DIR):
        print(f"[-] ERROR: Cannot find audio folder at {TRAIN_AUDIO_DIR}")
        return

    real_count = 0
    fake_count = 0

    print("[*] Reading protocol file and sorting audio...")
    
    with open(PROTOCOL_FILE, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Stop if we hit our limit for both categories
        if real_count >= LIMIT_PER_CLASS and fake_count >= LIMIT_PER_CLASS:
            break

        # ASVspoof line format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        audio_filename = parts[1] + ".flac"
        label = parts[4] # 'bonafide' or 'spoof'
        
        source_path = os.path.join(TRAIN_AUDIO_DIR, audio_filename)
        
        if not os.path.exists(source_path):
            continue

        if label == "bonafide" and real_count < LIMIT_PER_CLASS:
            shutil.copy(source_path, os.path.join(REAL_DIR, audio_filename))
            real_count += 1
            if real_count % 500 == 0:
                print(f"    -> Copied {real_count} REAL files...")
                
        elif label == "spoof" and fake_count < LIMIT_PER_CLASS:
            shutil.copy(source_path, os.path.join(FAKE_DIR, audio_filename))
            fake_count += 1
            if fake_count % 500 == 0:
                print(f"    -> Copied {fake_count} FAKE files...")

    print("\n[+] Sorting Complete!")
    print(f"    Total Real (Bonafide): {real_count}")
    print(f"    Total Fake (Spoof): {fake_count}")

if __name__ == "__main__":
    setup_directories()
    sort_asvspoof()