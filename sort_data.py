import os
import shutil

# Set paths dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_FILE = os.path.join(BASE_DIR, 'LA', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
AUDIO_DIR = os.path.join(BASE_DIR, 'LA', 'ASVspoof2019_LA_train', 'flac')

TARGET_REAL = os.path.join(BASE_DIR, 'dataset', 'real')
TARGET_FAKE = os.path.join(BASE_DIR, 'dataset', 'fake')

os.makedirs(TARGET_REAL, exist_ok=True)
os.makedirs(TARGET_FAKE, exist_ok=True)

print("[*] Starting ASVspoof 2019 Data Sorter...")

if not os.path.exists(PROTOCOL_FILE):
    print(f"[-] ERROR: Cannot find protocol file at {PROTOCOL_FILE}")
    print("Please ensure you copied the 'LA' folder directly into your Deepguard_Project directory.")
    exit()

with open(PROTOCOL_FILE, 'r') as f:
    lines = f.readlines()

real_count = 0
fake_count = 0

print("[*] Copying audio files. This might take a minute...")

for line in lines:
    parts = line.strip().split()
    if len(parts) < 5:
        continue
        
    filename = parts[1] + ".flac"
    label = parts[4] # 'bonafide' or 'spoof'
    
    src = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(src):
        continue
        
    if label == "bonafide":
        shutil.copy2(src, os.path.join(TARGET_REAL, filename))
        real_count += 1
    elif label == "spoof":
        shutil.copy2(src, os.path.join(TARGET_FAKE, filename))
        fake_count += 1

print(f"[+] Success! Copied {real_count} real files to dataset/real.")
print(f"[+] Success! Copied {fake_count} fake files to dataset/fake.")
print("[+] You can now run python train.py!")