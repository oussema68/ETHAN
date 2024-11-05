

import os
import random
from audio_processor import AudioProcessor

class AudioReconstructor:
    def __init__(self, dataset, sample_rate=16000):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)

    def reconstruct_random_audio(self):
        # Select a random sample index from the dataset
        sample_index = random.randint(0, len(self.dataset['test_0']) - 1)
        input_values = self.dataset['test_0'][sample_index]['input_values']

        # Define the WAV file path
        wav_filename = f'reconstructed_audio_{sample_index}.wav'
        wav_filepath = os.path.join('output', 'wav_files', wav_filename)

        # Reconstruct audio only if it doesn't already exist
        if not os.path.exists(wav_filepath):
            print(f"Reconstructing and saving audio for sample {sample_index}...")
            self.audio_processor.save_as_wav(input_values, filename=wav_filepath)
            print(f"WAV file created: {wav_filepath}")
        else:
            print(f"WAV file already exists for sample {sample_index}: {wav_filepath}")
