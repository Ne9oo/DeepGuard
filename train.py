import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = 'dataset'
EPOCHS = 10        # Transfer learning requires fewer epochs
BATCH_SIZE = 4     # Small batch size to manage RAM with long audio files

def extract_dynamic_features(filepath):
    """Generates a variable-width 3-channel Mel-spectrogram."""
    try:
        y, sr = librosa.load(filepath, sr=22050)
        if len(y) < 1000:
            return None

        # Generate Mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize pixel values between 0 and 1
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
        
        # Stack into 3 channels (RGB) to match MobileNetV2
        S_3channel = np.stack((S_norm,) * 3, axis=-1)
        return S_3channel
    except Exception as e:
        print(f"[-] Error extracting features from {filepath}: {e}")
        return None

def load_data():
    """Reads full audio files and labels them (0 = Real, 1 = Fake)."""
    features, labels = [], []
    valid_exts = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.mpeg')

    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(valid_exts):
                    print(f"[*] Processing {folder} audio: {file}")
                    data = extract_dynamic_features(os.path.join(folder_path, file))
                    if data is not None:
                        features.append(data)
                        labels.append(label)
                        
    return features, np.array(labels)

def pad_features(features):
    """Pads variable-length spectrograms to the same width so Keras can batch them."""
    max_len = max(f.shape[1] for f in features)
    padded = []
    for f in features:
        pad_width = max_len - f.shape[1]
        # Pad only the time dimension (axis 1)
        padded_f = np.pad(f, pad_width=((0,0), (0, pad_width), (0,0)), mode='constant')
        padded.append(padded_f)
    return np.array(padded)

def build_transfer_learning_model():
    """Builds a CNN using a pre-trained MobileNetV2 base."""
    # FIX: (None, None, 3) bypasses the Keras padding bug while keeping dynamic lengths
    base_model = MobileNetV2(input_shape=(None, None, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the pre-trained weights

    model = Sequential([
        base_model,
        # GlobalAveragePooling condenses any time-width into a fixed vector
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("[*] Starting DeepGuard Transfer Learning Process...")
    
    features_list, y = load_data()
    
    if len(features_list) < 2:
        print("[-] ERROR: You need at least 1 real and 1 fake audio file in the dataset folders.")
        exit()
        
    print("[*] Padding spectrograms for batch training...")
    X = pad_features(features_list)
    print(f"[+] Final Training Tensor Shape: {X.shape}")
    
    # Adjust test split based on small dataset sizes
    test_size = 0.2 if len(X) >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = build_transfer_learning_model()
    model.summary()
    
    print("\n[*] Training the Neural Network...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n[+] Final Model Test Accuracy: {test_accuracy * 100:.2f}%")
    
    model.save('deepguard_cnn.h5')
    print("[+] Model saved successfully as 'deepguard_cnn.h5'. You can now start app.py!")