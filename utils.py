import numpy as np
import h5py
import os
from scipy.io import wavfile
import librosa
import librosa.filters

def read_hdf5(hdf5_name, hdf5_path='feats'):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        raise Exception(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        raise Exception(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                #raise Exception("Dataset in hdf5 file already exists. "
                #                "recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                #logging.error("Dataset in hdf5 file already exists. "
                #              "if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

fmin=80
fmax=7600
hop_size=240
win_length=1024
fft_size=1024
highpass_cutoff=70.0
min_level_db=-100
sample_rate=24000
window='hann'
silence_threshold=2
num_mels=80
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _build_mel_basis():
    if fmax is not None:
        assert fmax <= sample_rate // 2
    return librosa.filters.mel(sample_rate, fft_size,
                               fmin=fmin, fmax=fmax,
                               n_mels=num_mels)
def _linear_to_mel(spectrogram):
    _mel_basis = None
    #global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)
def _stft(y, pad_mode="constant"):
    # use constant padding (defaults to zeros) instead of reflection padding
    return librosa.stft(y=y, n_fft=fft_size, hop_length=hop_size,
                        win_length=win_length, window=window,
                        pad_mode=pad_mode)
def logmelspectrogram(y, pad_mode="reflect"):
    """Same log-melspectrogram computation as espnet
    https://github.com/espnet/espnet
    from espnet.transform.spectrogram import logmelspectrogram
    """
    D = _stft(y, pad_mode=pad_mode)
    S = _linear_to_mel(np.abs(D))
    S = np.log10(np.maximum(S, 1e-10))
    return S
def trim(quantized):
    start, end = start_and_end_indices(quantized, silence_threshold)
    return quantized[start:end]
def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    if sr != sample_rate:
        x = librosa.resample(x, sr, sample_rate)
    x = np.clip(x, -1.0, 1.0)

    return x

def low_cut_filter(x, fs, cutoff=70):
    """APPLY LOW CUT FILTER.

    https://github.com/kan-bayashi/PytorchWaveNetVocoder

    Args:
        x (ndarray): Waveform sequence.
        fs (int): Sampling frequency.
        cutoff (float): Cutoff frequency of low cut filter.
    Return:
        ndarray: Low cut filtered waveform sequence.
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    from scipy.signal import firwin, lfilter

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x
