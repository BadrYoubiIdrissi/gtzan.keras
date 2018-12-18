import numpy as np
from gtzan import to_stft
from gtzan import splitsongs
from keras.utils import to_categorical
import numpy as np
from keras.utils import Sequence
import librosa




class DataGenerator(Sequence):

    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        Y = []

        for i in range(self.batch_size):
            x, y = self.load_from_filename(batch_x[i], batch_y[i], 512) 
            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)
    
    def load_from_filename(self, filename, label, nperseg=512, song_samples=660000):
        signal, fs = librosa.load(filename)
        signal = signal[:song_samples]
        signals, y = splitsongs(signal, label)
        specs = to_stft(signals, fs, nperseg)
        return specs, to_categorical(np.array(y), num_classes=10, dtype='int32')

def generator(X,y,steps_per_epoch, fs, nperseg):
    while True:
        for i in range(steps_per_epoch):
            start = (X.shape[0] // steps_per_epoch)*i
            if i == steps_per_epoch - 1:
                end = X.shape[0]
            else:
                end  = (X.shape[0] // steps_per_epoch)*(i+1)+1
            yield to_stft(X[start:end], fs, nperseg), y[start:end]


