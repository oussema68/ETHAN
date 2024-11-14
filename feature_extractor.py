import librosa
import numpy as np
import os

class FeatureExtractor:
    def extract_features(self, x, sr):
        zero_crossings = librosa.zero_crossings(x, pad=False)
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
        mfccs = librosa.feature.mfcc(y=x, sr=sr)
        chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=512)
        return mfccs, chromagram, zero_crossings, rolloff

    def save_features(self, mfccs, chromagram, i, partition="train", base_dir='output'):
        # Define the directory paths for each feature type, organized by partition
        mfcc_dir = os.path.join(base_dir, partition, 'mfccs')
        chroma_dir = os.path.join(base_dir, partition, 'chromas')

        # Create directories if they don't exist
        os.makedirs(mfcc_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

        # Save features in .npy format
        mfcc_file = os.path.join(mfcc_dir, f'mfccs_{i}.npy')
        chroma_file = os.path.join(chroma_dir, f'chroma_{i}.npy')

        np.save(mfcc_file, mfccs)
        np.save(chroma_file, chromagram)

        #print(f"Features saved: {mfcc_file}, {chroma_file}")

# Note: Zero-crossings and spectral rolloff could be saved later if needed but 
# are often less useful as standalone features for the current context.
