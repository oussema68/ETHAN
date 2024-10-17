import librosa
import numpy as np

class FeatureExtractor:
    def extract_features(self, x, sr):
        zero_crossings = librosa.zero_crossings(x, pad=False)
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
        mfccs = librosa.feature.mfcc(y=x, sr=sr)
        chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=512)
        return mfccs, chromagram, zero_crossings, rolloff

    def save_features(self, mfccs, chromagram, i, base_dir='output'):
        np.save(f'{base_dir}/mfccs/mfccs_{i}.npy', mfccs)
        np.save(f'{base_dir}/chromas/chroma_{i}.npy', chromagram)
        #print(f"Features saved: mfccs_{i}.npy, chroma_{i}.npy")
