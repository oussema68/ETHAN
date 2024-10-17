import os
from dataset_handler import DatasetHandler
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from tqdm import tqdm  # For progress visualization

def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)

def file_exists(file_path):
    """Helper function to check if a file exists."""
    return os.path.exists(file_path)

def process_dataset():
    dataset_handler = DatasetHandler()
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    
    dataset = dataset_handler.load_data()
    sample_rate = 16000
    create_directories()

    dataset_length = len(dataset['test_0'])

    # Progress bars for overall tasks
    with tqdm(total=dataset_length, desc="Reconstructing Audio", unit="file") as audio_bar, \
         tqdm(total=dataset_length, desc="Extracting Features", unit="file") as feature_bar:

        for i in range(dataset_length):
            input_values = dataset['test_0'][i]['input_values']
            wav_filename = f'reconstructed_audio_{i}.wav'
            wav_filepath = os.path.join('output', 'wav_files', wav_filename)

            mfcc_filename = f'mfcc_{i}.npy'
            chroma_filename = f'chroma_{i}.npy'
            mfcc_filepath = os.path.join('output', 'mfccs', mfcc_filename)
            chroma_filepath = os.path.join('output', 'chromas', chroma_filename)

            # Check if audio file already exists, if not, reconstruct it
            if not file_exists(wav_filepath):
                audio_processor.save_as_wav(input_values, filename=wav_filepath)
            
            # Update audio reconstruction progress
            audio_bar.update(1)

            # Reload the audio for feature extraction
            x, sr = audio_processor.reload_audio(wav_filepath)

            # Check if features already exist, if not, extract and save them
            if not file_exists(mfcc_filepath) or not file_exists(chroma_filepath):
                mfccs, chromagram, zero_crossings, rolloff = feature_extractor.extract_features(x, sr)
                feature_extractor.save_features(mfccs, chromagram, i)

            # Update feature extraction progress
            feature_bar.update(1)

if __name__ == "__main__":
    process_dataset()
