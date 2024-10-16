import matplotlib.pyplot as plt
import librosa.display

class Visualizer:
    def visualize_audio_features(self, input_values):
        plt.figure(figsize=(14, 5))
        plt.plot(input_values)
        plt.title("Audio Feature Vector Visualization")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_waveform(self, x, sr):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(x, sr=sr)
        plt.title("Waveform Visualization")
        plt.show()

    def plot_spectrogram(self, x, sr):
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.title("Spectrogram")
        plt.show()
