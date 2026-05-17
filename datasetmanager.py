import os
import csv
import numpy as np

import DCT
import melFilterBank


DATASET_FILE = "mfcc_dataset.csv"


# EXTRACT MFCC FEATURE VECTOR (13-D)

def extract_feature_vector(start_frame, frames_to_compute):

    # Select FFT frames dynamically
    FFTbins, nextFrame = melFilterBank.selectiveFrequencyBins(
        start=start_frame,
        framesToCompute=frames_to_compute
    )

    # Compute MFCCs
    MFCCS = DCT.get_mfcc(FFTbins)

    # Mean across time frames
    feature_vector = MFCCS.mean(axis=0)

    return feature_vector.astype(np.float32)


# CREATE DATASET FILE

def initialize_dataset():

    if os.path.exists(DATASET_FILE):
        return

    header = ["label"] + [f"mfcc_{i+1}" for i in range(13)]

    with open(DATASET_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(header)

    print("Dataset created successfully.")


# ADD SAMPLE

def add_sample(label, start_frame, frames_to_compute):

    feature_vector = extract_feature_vector(
        start_frame,
        frames_to_compute
    )

    row = [label] + feature_vector.tolist()

    with open(DATASET_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(row)

    print("Sample added successfully.")


# MAIN

if __name__ == "__main__":

    initialize_dataset()

    label = input("Enter label: ")

    start_frame = int(input("Enter start frame: "))

    frames_to_compute = int(
        input("Enter number of frames: ")
    )

    add_sample(
        label,
        start_frame,
        frames_to_compute
    )