import os
from dataset_handler import DatasetHandler
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from visualizer import Visualizer
from tqdm import tqdm  # For progress visualization

def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)

def file_exists(file_path):
    """Helper function to check if a file exists."""
    return os.path.exists(file_path)

def process_dataset(batch_size=1000):
    dataset_handler = DatasetHandler()
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    visualizer = Visualizer()
    
    dataset = dataset_handler.load_data()
    sample_rate = 16000
    create_directories()

    # Total dataset length
    dataset_length = len(dataset['test_0'])
    
    # Progress bar
    with tqdm(total=dataset_length, desc="Processing dataset") as pbar:
        for i in range(dataset_length):
            input_values = dataset['test_0'][i]['input_values']
            wav_filename = f'reconstructed_audio_{i}.wav'
            wav_filepath = os.path.join('output', 'wav_files', wav_filename)

            mfcc_filename = f'mfcc_{i}.npy'
            chroma_filename = f'chroma_{i}.npy'
            mfcc_filepath = os.path.join('output', 'mfccs', mfcc_filename)
            chroma_filepath = os.path.join('output', 'chromas', chroma_filename)

            # Check if both audio and feature files exist
            if file_exists(wav_filepath) and file_exists(mfcc_filepath) and file_exists(chroma_filepath):
                print(f"Audio and features for audio {i} already exist. Skipping.")
            else:
                # Process and save audio if it doesn't exist
                if not file_exists(wav_filepath):
                    print(f"Processing and saving audio file {wav_filename}...")
                    audio_processor.save_as_wav(input_values, filename=wav_filepath)
                
                # Reload the audio (if needed for features extraction)
                x, sr = audio_processor.reload_audio(wav_filepath)

                # Preprocess the audio
                x_normalized, y_filt = audio_processor.preprocess_audio(x)

                # Extract and save features if they don't exist
                if not file_exists(mfcc_filepath) or not file_exists(chroma_filepath):
                    print(f"Extracting and saving features for audio {i}...")
                    mfccs, chromagram, zero_crossings, rolloff = feature_extractor.extract_features(x, sr)
                    feature_extractor.save_features(mfccs, chromagram, i)

            # Update progress bar
            pbar.update(1)

if __name__ == "__main__":
    process_dataset()
