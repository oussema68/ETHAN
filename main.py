import os
from dataset_handler import DatasetHandler
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from tqdm import tqdm  # For progress visualization
from completion_checker import check_processing_completion


def create_directories(base_dir='output'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'wav_files'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'mfccs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'chromas'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'markers'), exist_ok=True)  # Directory for marker files

def process_dataset():
    dataset_handler = DatasetHandler()
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    
    dataset = dataset_handler.load_data()
    sample_rate = 16000
    create_directories()

    dataset_length = len(dataset['test_0'])

    # Path to the global marker file that indicates the entire dataset has been processed
    dataset_marker_filepath = os.path.join('output', 'markers', 'dataset_processed.txt')

    # Check if the dataset marker file exists; if yes, skip the entire processing
    if os.path.exists(dataset_marker_filepath):
        print("Dataset has already been processed. Skipping...")
        return

    # Otherwise, proceed to process the dataset
    with tqdm(total=dataset_length, desc="Processing Audio and Features", unit="file") as progress_bar:

        for i in range(dataset_length):
            wav_filename = f'reconstructed_audio_{i}.wav'
            wav_filepath = os.path.join('output', 'wav_files', wav_filename)

            mfcc_filepath = os.path.join('output', 'mfccs', f'mfcc_{i}.npy')
            chroma_filepath = os.path.join('output', 'chromas', f'chroma_{i}.npy')

            input_values = dataset['test_0'][i]['input_values']

            # Reconstruct audio if necessary
            if not os.path.exists(wav_filepath):
                audio_processor.save_as_wav(input_values, filename=wav_filepath)

            # Reload the audio for feature extraction
            x, sr = audio_processor.reload_audio(wav_filepath)

            # Extract and save features if necessary
            if not os.path.exists(mfcc_filepath) or not os.path.exists(chroma_filepath):
                mfccs, chromagram, zero_crossings, rolloff = feature_extractor.extract_features(x, sr)
                feature_extractor.save_features(mfccs, chromagram, i)

            # Update progress bar after processing
            progress_bar.update(1)

    # Once all files are processed, create the dataset marker file
    with open(dataset_marker_filepath, 'w') as dataset_marker_file:
        dataset_marker_file.write("Dataset processed successfully.")

    print("Processing complete. Dataset marker file created.")

if __name__ == "__main__":
    dataset_marker_filepath = os.path.join('output', 'markers', 'dataset_processed.txt')

    if check_processing_completion(dataset_marker_filepath):
        # Skip processing if marker file exists
        pass
    else:
        # Start processing the dataset
        process_dataset()
