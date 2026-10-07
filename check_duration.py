import os
import librosa
from datetime import timedelta

# Define the paths based on your folder structure
DATASET_DIR = "dataset"
CATEGORIES = ["real", "fake"]

def get_directory_duration(directory):
    total_duration = 0.0
    file_count = 0
    
    if not os.path.exists(directory):
        print(f"[-] Directory not found: {directory}")
        return 0.0, 0

    print(f"Scanning '{directory}'...")
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip subdirectories if any exist
        if os.path.isfile(filepath):
            try:
                # librosa.get_duration is highly efficient for checking file length
                duration = librosa.get_duration(path=filepath)
                total_duration += duration
                file_count += 1
            except Exception as e:
                print(f"  [!] Skipping {filename} - Could not read audio length. Error: {e}")
                
    return total_duration, file_count

if __name__ == "__main__":
    print("====================================")
    print("  DEEPGUARD DATASET DURATION CHECK  ")
    print("====================================\n")
    
    grand_total_duration = 0.0
    grand_total_files = 0
    
    for category in CATEGORIES:
        cat_path = os.path.join(DATASET_DIR, category)
        duration_sec, count = get_directory_duration(cat_path)
        
        grand_total_duration += duration_sec
        grand_total_files += count
        
        # Convert seconds to a readable HH:MM:SS format
        td = timedelta(seconds=int(duration_sec))
        
        print(f"\n[{category.upper()}] Audio Data:")
        print(f" -> Total Files: {count}")
        print(f" -> Total Duration: {td} (HH:MM:SS)\n")
        print("-" * 36)
        
    grand_td = timedelta(seconds=int(grand_total_duration))
    
    print("\n====================================")
    print("        GRAND TOTAL SUMMARY         ")
    print("====================================")
    print(f" -> Total Audio Files: {grand_total_files}")
    print(f" -> Total Dataset Duration: {grand_td} (HH:MM:SS)")
    print("====================================\n")