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

def process_dataset():
    dataset_handler = DatasetHandler()
    #feature_extractor = FeatureExtractor()
    
    # Load dataset and set up output directories
    dataset = dataset_handler.load_data()
    create_directories()
    
    # Path to the global marker file indicating the entire dataset has been processed
    dataset_marker_filepath = os.path.join('output', 'markers', 'dataset_processed.txt')

    # Check if the dataset has already been processed
    if os.path.exists(dataset_marker_filepath):
        print("Dataset has already been processed. Skipping...")
        return

    # Process the dataset (main logic)
    for i, sample in enumerate(tqdm(dataset['test_0'], desc="Processing Audio and Features", unit="file")):
        # Perform feature extraction etc. (skip code for brevity)
        pass

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
    process_dataset()
