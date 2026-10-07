"""DeepGuard detector: PyTorch CNN trained on ASVspoof 2019 LA.

Put this file and best_model.pt next to app.py.
Usage inside app.py:
    from deepguard_model import load_detector, predict_fake_prob
    detector = load_detector(os.path.join(app.root_path, 'best_model.pt'))
    fake_prob = predict_fake_prob(detector, y, sr)   # 0.0 = real, 1.0 = fake
"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio

SR = 16000
N_SAMPLES = 4 * SR  # 4-second windows, same as training


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    """Must stay identical to the CNN class in train.py, or the weights won't load."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=1024, hop_length=256, n_mels=80)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(10)
        self.time_mask = torchaudio.transforms.TimeMasking(20)
        self.features = nn.Sequential(
            _block(1, 16), _block(16, 32), _block(32, 64), _block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, wav):
        with torch.no_grad():
            x = self.to_db(self.mel(wav)).unsqueeze(1)
            x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5)
            if self.training:
                x = self.time_mask(self.freq_mask(x))
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def load_detector(weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def _make_windows(y):
    if len(y) < N_SAMPLES:
        y = np.tile(y, int(np.ceil(N_SAMPLES / len(y))))
    starts = list(range(0, len(y) - N_SAMPLES + 1, N_SAMPLES))
    if (len(y) - N_SAMPLES) % N_SAMPLES != 0:
        starts.append(len(y) - N_SAMPLES)
    windows = []
    for s in starts:
        w = y[s:s + N_SAMPLES]
        windows.append(w / (np.abs(w).max() + 1e-9))
    return np.stack(windows).astype(np.float32)


@torch.no_grad()
def predict_fake_prob(detector, y, sr):
    """y = mono float waveform (as returned by librosa.load). Returns P(fake) in [0, 1]."""
    model, device = detector
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < SR // 2:
        raise ValueError("Audio is too short to analyse (under 0.5 seconds).")
    windows = _make_windows(y)
    p_real = []
    for i in range(0, len(windows), 32):  # batches, so 5-minute files don't exhaust memory
        batch = torch.from_numpy(windows[i:i + 32]).to(device)
        p_real.append(torch.softmax(model(batch), dim=1)[:, 1].cpu())
    return 1.0 - float(torch.cat(p_real).mean())