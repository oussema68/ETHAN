import os
from datasets import load_dataset

class DatasetHandler:
    """
    Class to handle loading datasets from Hugging Face, ensuring that
    the dataset is not re-downloaded if it already exists locally.
    """
    
    def __init__(self, dataset_name="willcai/wav2vec2_common_voice_accents_3", data_dir='./data'):
        """
        Initialize the DatasetHandler with the dataset name and data directory.
        
        Parameters:
        dataset_name (str): The name of the dataset to load from Hugging Face.
        data_dir (str): The directory where the dataset is stored locally.
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.data_dir = data_dir

    def dataset_exists(self):
        """
        Check if the dataset files exist in the specified data directory.

        Returns:
        bool: True if the dataset exists, False otherwise.
        """
        # Check if the data directory exists and contains files
        return os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0

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
            self.dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
        else:
            try:
                print("Downloading dataset from Hugging Face...")
                self.dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)
            except Exception as e:
                raise ValueError(f"Error loading dataset '{self.dataset_name}': {e}")

        return self.dataset
