#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import torch
from transformers import pipeline
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf
from sklearn.preprocessing import minmax_scale
from datasets import load_dataset

# Load the dataset
def load_data():
    ds = load_dataset("willcai/wav2vec2_common_voice_accents_3")
    return ds

# Visualize audio features
def visualize_audio_features(input_values):
    plt.figure(figsize=(14, 5))
    plt.plot(input_values)  # Plot the feature vector directly
    plt.title("Audio Feature Vector Visualization")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Convert features to a .wav file
def save_as_wav(input_values, sample_rate, filename='reconstructed_audio.wav'):
    input_values = np.array(input_values)  # Convert to NumPy array if it's not already
    # Normalize the values to be within the range -1 to 1
    input_values = np.interp(input_values, (input_values.min(), input_values.max()), (-1, 1))
    sf.write(filename, input_values, sample_rate)
    print(f"WAV file created: {filename}")

# Reload the .wav file
def reload_audio(filename, sr=44000):
    x, sr = librosa.load(filename, sr=sr)
    return x, sr

# Plot the waveform
def plot_waveform(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    plt.title("Waveform Visualization")
    plt.show()

# Visualize the spectrogram
def plot_spectrogram(x, sr):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()

# Preprocess the audio
def preprocess_audio(x):
    # Normalization
    x_normalized = minmax_scale(x)
    # Pre-emphasis
    y_filt = librosa.effects.preemphasis(x)
    return x_normalized, y_filt

# Extract features
def extract_features(x, sr):
    zero_crossings = librosa.zero_crossings(x, pad=False)
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=512)
    
    return mfccs, chromagram, zero_crossings, rolloff

# Create directories for saving files
def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)

# Save extracted features
def save_features(mfccs, chromagram, i, base_dir='output'):
    np.save(os.path.join(base_dir, 'mfccs', f'mfccs_{i}.npy'), mfccs)       # Save MFCCs
    np.save(os.path.join(base_dir, 'chromas', f'chroma_{i}.npy'), chromagram)  # Save Chroma Frequencies
    print(f"Features saved: mfccs_{i}.npy, chroma_{i}.npy")

# Process the entire dataset
def process_dataset(ds):
    sample_rate = 16000  # Sample rate for saving .wav files
    create_directories()  # Create output directories
    
    for i in range(len(ds['test_0'])):  # Iterate through the dataset
        input_values = ds['test_0'][i]['input_values']
        visualize_audio_features(input_values)
        
        # Save the audio as a .wav file
        wav_filename = f'reconstructed_audio_{i}.wav'
        save_as_wav(input_values, sample_rate, filename=os.path.join('output', 'wav_files', wav_filename))
        
        # Reload the audio
        audio_data = os.path.join('output', 'wav_files', wav_filename)
        x, sr = reload_audio(audio_data, sr=sample_rate)
        
        # Plot waveform and spectrogram
        plot_waveform(x, sr)
        plot_spectrogram(x, sr)
        
        # Preprocess audio
        x_normalized, y_filt = preprocess_audio(x)
        
        # Extract features
        mfccs, chromagram, zero_crossings, rolloff = extract_features(x, sr)
        
        # Save extracted features
        save_features(mfccs, chromagram, i)

# Main execution
if __name__ == "__main__":
    dataset = load_data()
    process_dataset(dataset)
