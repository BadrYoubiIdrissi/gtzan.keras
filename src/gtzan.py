import os
import gc
import logging
import argparse
from datetime import datetime
from collections import OrderedDict

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Audio processing and DL frameworks 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.models import load_model

from gtzan import build_model, read_test_data, read_data
from dataGeneration import generator

from gtzan.constants import *

def main(args):
    exec_mode = ['train', 'test']
    exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

    # Validate arguments
    if args.type not in exec_mode:
        raise ValueError("Invalid type parameter. Should be 'train' or 'test'.")

    # Start
    if args.type == 'train':
        # Check if the directory path to GTZAN files was inputed
        if not args.directory:
            raise ValueError("File path to model should be passed in test mode.")

        # Create directory to save logs
        try:
            os.mkdir('../logs/{}'.format(exec_time))
        except FileExistsError:
            # If the directory already exists
            pass 

        # Read the files to memory and split into train test
        X, y, fs = read_data(args.directory, genres, song_samples)

        # Transform to a 3-channel image
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

        # Training step
        input_shape = (257,259,2)
        cnn = build_model(input_shape, num_genres)
        cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

        cnn.fit_generator(generator(X_train,y_train,100, fs, nperseg=512), steps_per_epoch=100, epochs=50, verbose=1)

        # Evaluate
        score = cnn.evaluate_generator(generator(X_test, y_test, 100, fs, nperseg=512), steps=100, verbose = 0)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

        # Save the model
        cnn.save('../models/{}.h5'.format(exec_time))

    else:
        # Check if the file path to the model was passed
        if not args.model:
            raise ValueError("File path to model should be passed in test mode.")

        # Check if was passed the music file
        if not args.song:
            raise ValueError("Song path should be passed in test mode.")

        model = load_model(args.model)
        X = read_test_data(args.song)
        y = model.predict(np.squeeze(np.stack((X,) * 3, -1)))
        print([genresRev[x] for x in np.argmax(y, axis=1)])



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    # Required arguments
    parser.add_argument('-t', '--type', help='train or test mode to execute', type=str, required=True)

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-d', '--directory', help='Path to the root directory with GTZAN files', type=str)
    parser.add_argument('-m', '--model', help='If choosed test, path to trained model', type=str)
    parser.add_argument('-s', '--song', help='If choosed test, path to song to classify', type=str)
    args = parser.parse_args()

    # Call the main function
    main(args)