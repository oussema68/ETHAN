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


    partitions = {'train_0': 'train_processed.txt', 'test_0': 'test_processed.txt'}

    for partition, marker_filename in partitions.items():
        partition_marker_filepath = os.path.join('output', 'markers', marker_filename)

        
        # Check if the partition has already been processed
        if os.path.exists(partition_marker_filepath):
            print(f"{partition} has already been processed. Skipping...")
            return
        else:
            # Initialize audio processor and feature extractor
            audio_processor = AudioProcessor()
            feature_extractor = FeatureExtractor()

            # Process each sample in the partition
            for i, sample in enumerate(tqdm(dataset[partition], desc=f"Processing {partition} Audio and Features", unit="file")):
                input_values = np.array(sample['input_values'])

                # Preprocess audio
                x_normalized, y_filt = audio_processor.preprocess_audio(input_values)

                # Skip empty or silent audio data
                if not np.any(x_normalized):
                    print(f"Skipping sample {i} in {partition} due to silence.")
                    continue

                try:
                    # Extract features
                    sr = 16000
                    mfccs, chromagram, _, _ = feature_extractor.extract_features(x_normalized, sr)

                    # Save features with partition directory
                    feature_extractor.save_features(mfccs, chromagram, i, partition=partition)

                except Exception as e:
                    print(f"Error processing sample {i} in {partition}: {e}")


            # Write a marker file after processing the partition
            with open(partition_marker_filepath, 'w') as f:
                f.write(f"{partition} processing completed.")






    

    
        

            


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
