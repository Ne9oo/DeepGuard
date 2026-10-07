import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

DATASET_PATH = 'dataset'
EPOCHS = 10
MAX_SAMPLES = 2000 
MAX_PAD_LEN = 216 

def extract_mfcc(filepath):
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Standardize time length
        if mfcc.shape[1] > MAX_PAD_LEN:
            mfcc = mfcc[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Z-Score Normalization (Prevents Sigmoid Saturation at 1.000000)
        mean = np.mean(mfcc)
        std = np.std(mfcc) + 1e-8
        mfcc_norm = (mfcc - mean) / std
            
        return np.expand_dims(mfcc_norm, axis=-1)
    except Exception:
        return None

def load_data():
    X, y = [], []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(('.flac', '.wav', '.ogg', '.mp3'))][:MAX_SAMPLES]
        print(f"[*] Loading {len(files)} files from '{folder}'...")
        
        for file in files:
            feats = extract_mfcc(os.path.join(folder_path, file))
            if feats is not None:
                X.append(feats)
                y.append(label)
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(40, 216, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("[*] Starting Balanced Normalized MFCC Training...")
    X, y = load_data()
    print(f"[+] Loaded {len(X)} files across classes: Real={np.sum(y==0)}, Fake={np.sum(y==1)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test))
    
    model.save('deepguard_cnn.h5')
    print("[+] Model saved successfully as 'deepguard_cnn.h5'.")