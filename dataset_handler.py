import os
from datasets import load_dataset

class DatasetHandler:
    def __init__(self, dataset_name="willcai/wav2vec2_common_voice_accents_3", data_dir='data'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir  # Directory where the dataset is cached
        self.dataset = None

    def dataset_exists(self):
        """Check if the dataset files exist in the specified directory."""
        if not os.path.exists(self.data_dir):
            return False
        return any(os.listdir(self.data_dir))  # Check if the directory is not empty

    def load_data(self):
        """
        Load the dataset from Hugging Face or from local storage if it already exists.
        
        Returns:
        dataset (DatasetDict): The loaded dataset.
        
        Raises:
        ValueError: If the dataset could not be loaded.
        """
        if self.dataset_exists():
            print("Loading dataset from local storage...")
            try:
                self.dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
            except Exception as e:
                print(f"Failed to load dataset from cache: {e}. Attempting to download again.")
                self.download_dataset()
        else:
            self.download_dataset()

        return self.dataset

    def download_dataset(self):
        """Download the dataset from Hugging Face."""
        try:
            print("Downloading dataset from Hugging Face...")
            self.dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
        except Exception as e:
            raise ValueError(f"Error downloading dataset '{self.dataset_name}': {e}")
