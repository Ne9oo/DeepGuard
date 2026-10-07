import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MAX_PAD_LEN = 216

def test_audio(filepath, model_path='deepguard_cnn.h5'):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"[-] Could not load model: {e}")
        return
        
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        if mfcc.shape[1] > MAX_PAD_LEN:
            mfcc = mfcc[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Apply identical Z-score normalization
        mean = np.mean(mfcc)
        std = np.std(mfcc) + 1e-8
        mfcc_norm = (mfcc - mean) / std
        
        features = np.expand_dims(np.expand_dims(mfcc_norm, axis=-1), axis=0)
        pred = model.predict(features, verbose=0)
        fake_prob = float(pred[0][0])
        
        print("\n" + "="*45)
        print(f"FILE: {filepath}")
        print(f"RAW SCORE (0.0=Real, 1.0=Fake): {fake_prob:.4f}")
        if fake_prob > 0.5:
            print(f"VERDICT: SYNTHETIC DEEPFAKE ({fake_prob*100:.1f}%)")
        else:
            print(f"VERDICT: AUTHENTIC HUMAN ({(1-fake_prob)*100:.1f}%)")
        print("="*45 + "\n")
        
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_cnn.py <audio_path>")
    else:
        test_audio(sys.argv[1])