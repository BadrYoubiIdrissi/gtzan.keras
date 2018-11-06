import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import scipy

"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

def stft(x, fs=22050, nperseg=1024):
    _, _, Zxx = scipy.signal.stft(x, fs=fs, nperseg=nperseg)
    return np.stack([20*np.log(abs(Zxx)),np.angle(Zxx)], axis=-1)

def istft(x, fs=22050, nperseg=1024):
    pass
"""
@description: Method to convert a list of songs to a np array of stft amplitude and phase
"""
def to_stft(songs, fs=22050, nperseg=1024):
    # Transformation function
    stftMap = lambda x: stft(x, fs=fs, nperseg=nperseg)

    # map transformation of input songs to stft
    tsongs = map(stftMap, songs)
    return np.array(list(tsongs))

"""
@description: Read audio files from folder
"""
def read_data(src_dir, genres, song_samples = 660000, nperseg = 1024, debug = True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []

    # Read files from the folders
    for x, _ in genres.items():
        folder = src_dir + x
        
        for _, _, files in os.walk(folder):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, fs = librosa.load(file_name)
                signal = signal[:song_samples]

                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))
                
                # Convert to dataset of spectograms/melspectograms
                signals, y = splitsongs(signal, genres[x])
                
                # Convert to "spec" representation
                specs = to_stft(signals, fs)
                
                # Save files
                arr_genres.extend(y)
                arr_specs.extend(specs)
                
    return np.array(arr_specs), to_categorical(np.array(arr_genres))

def read_test_data(src_dir,song_samples = 660000,nperseg = 1024):
    signal, _ = librosa.load(src_dir)
    signal = signal[:song_samples]
    signals, fs = splitsongs(signal, 0)
    specs = to_stft(signals, fs, nperseg)
    return np.array(specs)