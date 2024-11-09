import os
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings globally

from dataset_handler import DatasetHandler
from feature_extractor import FeatureExtractor
from tqdm import tqdm
import numpy as np
#from completion_checker import check_processing_completion
from audio_reconstructor import AudioReconstructor
from audio_processor import AudioProcessor

def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'markers'), exist_ok=True)  # Directory for marker files

def process_dataset(dataset):
    create_directories()  # Ensures output directories are set up
    dataset_marker_filepath = os.path.join('output', 'markers', 'dataset_processed.txt')

    # Check if the dataset has already been processed
    if os.path.exists(dataset_marker_filepath):
        print("Dataset has already been processed. Skipping...")
        return
    else:
        # Initialize audio processor
        audio_processor = AudioProcessor()

        # Initialize feature extractor
        feature_extractor = FeatureExtractor()

        # Process each audio sample
       
        for i, sample in enumerate(tqdm(dataset['test_0'], desc="Processing Audio and Features", unit="file")):
            input_values = np.array(sample['input_values'])

            # Preprocess audio
            x_normalized, y_filt = audio_processor.preprocess_audio(input_values)

            # Skip empty or silent audio data
            if not np.any(x_normalized):
                print(f"Skipping sample {i} due to silence.")
                continue
            
            try:
                # Extract features only if audio is valid
                sr = 16000
                mfccs, chromagram, _, _ = feature_extractor.extract_features(x_normalized, sr)
                feature_extractor.save_features(mfccs, chromagram, i)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")

        # After processing all samples, create the marker file
        with open(dataset_marker_filepath, 'w') as marker_file:
            marker_file.write("Dataset processed successfully.")



            


if __name__ == "__main__":
    dataset_handler = DatasetHandler()
    dataset = dataset_handler.load_data()

    # Ask user if they want to reconstruct a random audio file
    choice = input("Would you like to reconstruct a random audio to test filtering? (y/n): ").strip().lower()
    if choice == 'y':
        reconstructor = AudioReconstructor(dataset)
        reconstructor.reconstruct_random_audio()

    # Run main processing logic
    process_dataset(dataset)
