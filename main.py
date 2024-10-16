import os
from dataset_handler import DatasetHandler
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from visualizer import Visualizer

def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)

def process_dataset():
    dataset_handler = DatasetHandler()
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    visualizer = Visualizer()
    
    dataset = dataset_handler.load_data()
    sample_rate = 16000
    create_directories()

    for i in range(len(dataset['test_0'])):
        input_values = dataset['test_0'][i]['input_values']
        wav_filename = f'reconstructed_audio_{i}.wav'
        
        # Save the audio as .wav
        audio_processor.save_as_wav(input_values, filename=os.path.join('output', 'wav_files', wav_filename))

        # Reload the audio
        x, sr = audio_processor.reload_audio(os.path.join('output', 'wav_files', wav_filename))

        # Optional: Plot waveform and spectrogram
        # visualizer.plot_waveform(x, sr)
        # visualizer.plot_spectrogram(x, sr)

        # Preprocess the audio
        x_normalized, y_filt = audio_processor.preprocess_audio(x)

        # Extract and save features
        mfccs, chromagram, zero_crossings, rolloff = feature_extractor.extract_features(x, sr)
        feature_extractor.save_features(mfccs, chromagram, i)

if __name__ == "__main__":
    process_dataset()
