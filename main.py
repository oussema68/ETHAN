import os
from dataset_handler import DatasetHandler
from feature_extractor import FeatureExtractor
from tqdm import tqdm
#from completion_checker import check_processing_completion
from audio_reconstructor import AudioReconstructor

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

    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    # Process each audio sample
    for i, sample in enumerate(tqdm(dataset['test_0'], desc="Processing Audio and Features", unit="file")):
        input_values = np.array(sample['input_values'])  # Convert input values to numpy array if necessary
        sr = 16000  # Sample rate (set to 16000 as an example, adjust as needed)

        # Extract features
        mfccs, chromagram, _, _ = feature_extractor.extract_features(input_values, sr)  # Ignore unwanted features

        # Save features using FeatureExtractor's save method
        feature_extractor.save_features(mfccs, chromagram, i)

    # Create marker file upon completion
    with open(dataset_marker_filepath, 'w') as dataset_marker_file:
        dataset_marker_file.write("Dataset processed successfully.")
    print("Processing complete.")


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
