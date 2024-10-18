import numpy as np
import soundfile as sf
import librosa
from sklearn.preprocessing import minmax_scale
import os

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def save_as_wav(self, input_values, filename='reconstructed_audio.wav'):
        # Check if the file already exists (to avoid repeating work)
        if not os.path.exists(filename):
            input_values = np.array(input_values)
            input_values = np.interp(input_values, (input_values.min(), input_values.max()), (-1, 1))
            sf.write(filename, input_values, self.sample_rate)
            print(f"WAV file created: {filename}")
        else:
            print(f"WAV file already exists: {filename}")

    def reload_audio(self, filename):
        if os.path.exists(filename):
            x, sr = librosa.load(filename, sr=self.sample_rate)
            return x, sr
        else:
            print(f"File not found: {filename}")
            return None, None

    def preprocess_audio(self, x):
        # Normalize and apply pre-emphasis
        x_normalized = minmax_scale(x)
        y_filt = librosa.effects.preemphasis(x)
        return x_normalized, y_filt
