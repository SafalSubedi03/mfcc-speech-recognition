'''
In this code first we will create a 2D mel spectrum vector where logarithmic energy form each triangular filter for each frame 
will me stored. Then DCT will be applied on each frame keeping the first 13 coefficients ignore the 0th coefficient.
Finally we will print the MFCCs for certain frames.

'''
import melFilterBank
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt


# Create mel spectrum from selected FFT bins
def compute_melspectrum(fftbins):

    total_frames = fftbins.shape[0]

    melspectrum = np.zeros(
        (total_frames, melFilterBank.NumberOfFilter),
        dtype=float
    )

    for i in range(total_frames):

        melspectrum[i,:] = melFilterBank.logmelspec(
            frame_number=i,
            fft_binsA=fftbins
        )

    return melspectrum


# Perform DCT
def performdct(melspec):

    total_frames = melspec.shape[0]

    cepstrum = np.zeros(
        (total_frames, melFilterBank.NumberOfFilter),
        dtype=float
    )

    for i in range(total_frames):

        cepstrum[i,:] = dct(
            melspec[i,:],
            type=2,
            norm='ortho'
        )

    return cepstrum[:,1:14]


# Main MFCC function
def get_mfcc(fftbins):

    melspectrum = compute_melspectrum(fftbins)

    MFCCS = performdct(melspectrum)

    return MFCCS


if __name__ == "__main__":

    # Select frames dynamically
    FFTbins, nextFrame = melFilterBank.selectiveFrequencyBins(
        start=0,
        framesToCompute=47
    )

    MFCCS = get_mfcc(FFTbins)

    # Normalize
    mfcc_norm = (
        (MFCCS - MFCCS.mean(axis=0))
        / MFCCS.std(axis=0)
    )

    feature_vector = MFCCS.mean(axis=0)

    frame_number = 40

    print(
        f"Unnormalized MFCC for frame {frame_number}:"
    )

    print(MFCCS[frame_number,:])

    print(f"\nNormalized MFCCs for frame {frame_number}:")
    print(mfcc_norm[frame_number,:])
    print("\nFeature Vector:")
    print(feature_vector)


    # Plot heatmaps

    time_axis = np.arange(
        mfcc_norm.shape[0]
    ) * melFilterBank.FFT.hop_length

    coeff_axis = np.arange(
        1,
        mfcc_norm.shape[1] + 1
    )

    fig, axs = plt.subplots(1,2, figsize=(12,6))


    # Normalized MFCC
    pcm = axs[0].pcolormesh(
        time_axis,
        coeff_axis,
        mfcc_norm.T,
        shading='auto',
        cmap='viridis'
    )

    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('MFCC Coefficient')
    axs[0].set_title('Normalized MFCCs')

    fig.colorbar(pcm, ax=axs[0])


    # Raw MFCC
    pcm = axs[1].pcolormesh(
        time_axis,
        coeff_axis,
        MFCCS.T,
        shading='auto',
        cmap='viridis'
    )

    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('MFCC Coefficient')
    axs[1].set_title('Raw MFCCs')

    fig.colorbar(pcm, ax=axs[1])

    plt.show()




