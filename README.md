# 🎶 Audio Processing and Feature Extraction Pipeline

This repository contains a comprehensive audio processing pipeline that handles datasets, processes audio files, extracts features, and saves the output to structured directories. The code is designed to be efficient, skipping previously processed datasets based on a completion marker.

## 🌟 Features

- **Dataset Handling**: Automatically loads datasets for processing.
- **Audio Processing**: Converts raw input data into `.wav` files.
- **Feature Extraction**: Extracts MFCCs, chromagrams, and other audio features.
- **Progress Visualization**: Shows real-time progress using `tqdm`.
- **Processing Skip Mechanism**: Skips datasets already processed by checking a marker file.

## 🚀 Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+**
- **tqdm** for progress bars
- **NumPy** and **LibROSA** for audio and feature extraction (part of `audio_processor` and `feature_extractor` modules)
- Any custom dependencies such as `dataset_handler` and `completion_checker` (defined in your project structure).

### 🛠 Setup

1. **Clone the Repository**:

   ```bash
   git clone (https://github.com/oussema68/ETHAN.git
   cd ETHAN
   ```

2. **Install Dependencies**:

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:

   Ensure that your dataset is correctly structured and the `DatasetHandler` class is set up to load it properly.

### 📂 Directory Structure

The output directories are automatically created when running the pipeline:

```
output/
├── wav_files/     # Processed .wav files
├── mfccs/         # MFCC features saved as .npy files
├── chromas/       # Chromagram features saved as .npy files
└── markers/       # Marker files to track processing completion
```

## 💻 Running the Pipeline

To run the full audio processing pipeline, simply execute:

```bash
python main.py
```

The pipeline will:

1. Load your dataset.
2. Process each audio sample, generating `.wav` files and extracting audio features such as MFCC and chromagrams.
3. Save the features in `.npy` format for easy loading and further analysis.
4. Track the processing with a marker file to avoid redundant re-processing.

### Example:

```bash
Processing Audio and Features: 100%|████████████████████| 100/100 [02:34<00:00,  2.57s/file]
Processing complete. Dataset marker file created.
```

### 💾 Dataset Re-Processing Control

The pipeline checks for the presence of a marker file (`dataset_processed.txt`) to determine whether the dataset has already been processed. If the marker file exists, the processing will be skipped:

```python
# If the marker file exists, skip processing
if os.path.exists(dataset_marker_filepath):
    print("Dataset has already been processed. Skipping...")
```

## 🧠 Key Functions

- **create_directories()**: Creates the output directories for storing the processed data.
- **process_dataset()**: Main function that orchestrates dataset loading, audio processing, and feature extraction.
- **save_as_wav()**: Saves input audio data as `.wav` files.
- **extract_features()**: Extracts MFCCs, chromagrams, and other relevant features from the audio.
- **check_processing_completion()**: Checks if the processing has already been completed by looking for the marker file.

## 🎛 Customization

You can modify the pipeline to extract additional features or adjust the dataset loading process to suit your needs. Customize the `FeatureExtractor` class to extend feature extraction based on your project requirements.

## 🌐 Deployment

This project can be deployed locally or in any cloud-based environment where Python is supported. If processing large datasets, ensure the system has sufficient CPU/GPU and memory resources.


---

# 🧩 Audio Processing Modules

This second repository contains helper modules (`dataset_handler`, `audio_processor`, `feature_extractor`, and others) used by the main audio processing pipeline.

## 🚀 Module Overview

### 1. **DatasetHandler**

Manages loading and handling datasets for audio processing. This module ensures the proper structure and format of datasets.

### 2. **AudioProcessor**

Handles audio file operations, including saving raw audio data as `.wav` files and reloading them for further processing.

#### Key Functions:
- `save_as_wav(input_values, filename)`: Saves input values as `.wav` files.
- `reload_audio(filepath)`: Reloads audio files for processing.

### 3. **FeatureExtractor**

Extracts audio features such as MFCC, chromagram, zero-crossing rate, and spectral roll-off.

#### Key Functions:
- `extract_features(x, sr)`: Extracts features from audio signal `x` with sample rate `sr`.
- `save_features(mfccs, chromagram, index)`: Saves extracted features as `.npy` files.

### 4. **CompletionChecker**

Provides utilities for checking whether dataset processing is complete, using marker files to skip already processed datasets.

---

## 💻 Running Unit Tests

To ensure the modules are functioning correctly, run the provided test suite:

```bash
python -m unittest discover tests/
```

## 📜 License

**This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.**

## ✨ Contributions

Fork this repository and contribute to improve the core modules!

---

Made with 💡 and 🎶 by Oussema Hammadi.
