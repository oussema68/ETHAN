import os

def check_processing_completion(marker_filepath):
    """Check if the dataset processing has been completed by looking for the marker file."""
    if os.path.exists(marker_filepath):
        print("Dataset has already been processed. Proceeding to the next step...")
        return True
    else:
        print("Dataset has not been processed yet. Starting processing...")
        return False
