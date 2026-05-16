# MFCCs & Speech Recognition From Scratch

This project demonstrates **MFCC (Mel-Frequency Cepstral Coefficients) computation from scratch** and uses a **PyTorch-based neural network** for speech recognition.

The system includes:
- Custom MFCC feature extraction pipeline (FFT → Mel Filter Bank → DCT)
- Dataset creation and management
- Neural network training using PyTorch
- Prediction of unknown audio samples

---

## Project Overview

Audio → FFT → Mel Filter Bank → Log Energies → DCT → MFCC (13-D) → Dataset Manager → PyTorch Model → Softmax Classification

---

## File Descriptions

### 1. FFT.py
Handles audio preprocessing:
- Loads WAV audio file
- Applies pre-emphasis filtering
- Frames signal into short segments
- Applies Hanning window
- Computes FFT per frame

---

### 2. melFilterBank.py
Performs frequency transformation:
- Converts Hz to Mel scale
- Builds triangular filter banks
- Computes log Mel energies
- Maps frequency bins to FFT spectrum

---

### 3. DCT.py
Extracts MFCC features:
- Applies Discrete Cosine Transform (DCT)
- Keeps 13 MFCC coefficients (excluding 0th)
- Produces frame-wise MFCC matrix
- Computes feature vector (mean across frames)

---

### 4. dataset_manager.py
Handles dataset creation:
- Extracts MFCC features from DCT.py
- Stores labeled samples in mfcc_dataset.csv
- Ensures consistent feature formatting for training

---

### 5. mfcc_train.ipynb
Jupyter Notebook for training:
- Loads dataset
- Encodes labels
- Normalizes MFCC features
- Trains PyTorch neural network
- Evaluates model performance
- Saves trained model and preprocessors

---

### 6. KNNForMFCCs.py (Legacy)
Old KNN-based classifier replaced by neural network for better accuracy and scalability.

---

## Model Inference

Audio → MFCC extraction → Normalization → PyTorch model → Softmax → Label

---

## Running the Project

1. Record audio using the provided recording script.
2. Add labeled data using dataset_manager.py.
3. Train the model using mfcc_train.ipynb.
4. Run inference on unknown audio samples using the trained model.

---

## System Requirements

- Python 3.11
- All dependencies are listed in requirements.txt

Install dependencies:

pip install -r requirements.txt

---

## Notes

- MFCC extraction is fully custom (no external MFCC libraries used).
- Consistent audio length improves model accuracy.
- Same preprocessing must be used during training and inference.
- Neural network replaces KNN for better generalization.

---

## License

This project is for educational and research purposes only.