from datasets import load_dataset

class DatasetHandler:
    def __init__(self, dataset_name="willcai/wav2vec2_common_voice_accents_3"):
        self.dataset_name = dataset_name
        self.dataset = None

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name)
        return self.dataset
