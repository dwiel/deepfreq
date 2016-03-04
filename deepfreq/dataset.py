import random

import numpy as np

import tqdm
from . import audio_file

# FILENAME = '/home/ubuntu/deepfreq/African Drum Music-wXV39pybgJU-sm.wav'
FILENAME = '/home/ubuntu/deepfreq/African Drum Music-wXV39pybgJU.wav'


def _filter_fft_slice():
    slice = np.zeros(fft.shape[0], dtype=bool)

    slice[4] = True
    slice[12] = True
    slice[18] = True

    return slice


def filter_fft(fft):
    slice = _filter_fft_slice()
    return fft[slice,:]


def expand_fft(fft, shape):
    expanded_fft = np.zeros(shape, dtype='complex')

    slice = _filter_fft_slice()
    expanded_fft[slice,:] = fft

    return expanded_fft


def complex_to_combined(fft):
    # return np.concatenate([np.abs(fft), np.angle(fft)])
    return np.abs(fft)


def combined_to_complex(combined):
    """
    mid = combined.shape[0]/2

    magnitude = combined[:mid,:]
    angle = combined[mid:,:]

    return magnitude * np.exp(1j * angle)
    """
    return combined + (0 * 1j)


def rescale_combined(combined):
    # mid = combined.shape[0]/2
    mid = combined.shape[0]

    scale = np.max(combined[:mid,:])

    combined[:mid,:] /= scale
    combined[mid:,:] /= np.pi

    # return combined, scale
    return combined


def dataset():
    # output shape is (time, freq_bins)
    return rescale_combined(complex_to_combined(
        filter_fft(
            audio_file.fft(
                audio_file.wav(
                    FILENAME
                )
            )
        )
    )).T


def sample_batch(dataset, length, batch_size):
    X = np.zeros((batch_size, length, dataset.shape[1]))
    Y = np.zeros((batch_size, length, dataset.shape[1]))

    if dataset.shape[0] < length:
        raise ValueError(
            'cant extract sequence length {} from sequence length {}'.format(
                dataset.shape[0], length
            )
        )

    for e in range(batch_size):
        i = random.randint(0, dataset.shape[0] - length - 1)

        X[e,:,:] = dataset[i:i+length,:]
        Y[e,:,:] = dataset[i+1:i+length+1,:]

    return X, Y
