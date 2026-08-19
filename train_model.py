import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = 'dataset'
MAX_PAD_LEN = 150  # Must match the length in app.py
EPOCHS = 15        # Number of training loops
BATCH_SIZE = 32

def extract_mfcc(filepath, max_pad_len=MAX_PAD_LEN):
    """Extracts MFCC from audio file and pads/truncates to a fixed length."""
    try:
        audio, sample_rate = librosa.load(filepath, sr=16000, duration=5.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

def load_data():
    """Reads the dataset folder and labels the data (0 = Real, 1 = Fake)."""
    features = []
    labels = []

    # 1. Load Real Audio (Label = 0)
    real_path = os.path.join(DATASET_PATH, 'real')
    if os.path.exists(real_path):
        for file in os.listdir(real_path):
            # Added .flac support here!
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                data = extract_mfcc(os.path.join(real_path, file))
                if data is not None:
                    features.append(data)
                    labels.append(0)  # 0 for Authentic

    # 2. Load Fake Audio (Label = 1)
    fake_path = os.path.join(DATASET_PATH, 'fake')
    if os.path.exists(fake_path):
        for file in os.listdir(fake_path):
            # Added .flac support here!
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                data = extract_mfcc(os.path.join(fake_path, file))
                if data is not None:
                    features.append(data)
                    labels.append(1)  # 1 for Deepfake

    return np.array(features), np.array(labels)

def build_cnn_model(input_shape):
    """Builds the Convolutional Neural Network architecture."""
    model = Sequential([
        # Layer 1: Feature Extraction
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Layer 2: Deep Feature Extraction
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Layer 3: Flattening and Decision Making
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output Layer: Binary Classification (0 to 1)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with Binary Crossentropy for Two-Class (Real/Fake) classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("[*] Starting DeepGuard CNN Training Process...")
    
    # 1. Load and process audio files
    print("[*] Loading and extracting MFCC features from dataset... This will take a few minutes!")
    X, y = load_data()
    
    if len(X) == 0:
        print("[-] ERROR: No audio data found. Please add .wav, .mp3, or .flac files to dataset/real and dataset/fake.")
        exit()
        
    print(f"[+] Extracted features from {len(X)} audio files.")
    
    # 2. Reshape data for CNN (Samples, 40, 150, 1)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # 3. Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build the model
    input_shape = (X.shape[1], X.shape[2], 1)
    model = build_cnn_model(input_shape)
    model.summary()
    
    # 5. Train the CNN
    print("\n[*] Training the Neural Network...")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    
    # 6. Evaluate accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n[+] Final Model Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # 7. Save the trained model for app.py to use
    model.save('deepguard_cnn.h5')
    print("[+] Model saved successfully as 'deepguard_cnn.h5'. You can now start app.py!")