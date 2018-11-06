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

import gtzan as gtz
from dataGeneration import generator

# Constants
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
genresRev = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 
          'country', 'pop', 'blues', 'reggae', 'rock']
num_genres = len(genres)

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
        """ try:
            os.mkdir('../logs/{}'.format(exec_time))
        except FileExistsError:
            # If the directory already exists
            pass """

        # Read the files to memory and split into train test
        # X, y = read_data(args.directory, genres, song_samples)
        
        # Load data
        X = np.load("../data.npy")
        y = np.load("../output.npy")



        # Transform to a 3-channel image
        # X_stack = np.squeeze(np.stack((X,) * 3, -1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

        # Histogram for train and test
        # values, count = np.unique(np.argmax(y_train, axis=1), return_counts=True)
        # plt.bar(values, count)

        # values, count = np.unique(np.argmax(y_test, axis=1), return_counts=True)
        # plt.bar(values, count)
        # plt.savefig('../logs/{}/histogram.png'.format(exec_time),
        #     format='png', bbox_inches='tight')

        # Training step
        input_shape = (128,129,3)
        cnn = gtz.build_model(input_shape, num_genres)
        cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

        # hist = cnn.fit(X_train, y_train,
        #         batch_size = 256,
        #         epochs = 50,
        #         verbose = 1,
        #         validation_data = (X_test, y_test))

        cnn.fit_generator(generator(X_train,y_train,100), steps_per_epoch=100, epochs=50, verbose=1)

        # Evaluate
        score = cnn.evaluate_generator(generator(X_test, y_test, 100), steps=100, verbose = 0)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

        # Plot graphs
        # save_history(hist, '../logs/{}/evaluate.png'.format(exec_time))

        # Save the confusion Matrix
        # preds = np.argmax(cnn.predict(X_test), axis = 1)
        # y_orig = np.argmax(y_test, axis = 1)
        # cm = confusion_matrix(preds, y_orig)

        #keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()
        #plot_confusion_matrix('../logs/{}/cm.png'.format(exec_time), cm, keys, normalize=True)

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
        X = gtz.read_test_data(args.song)
        y = model.predict(np.squeeze(np.stack((X,) * 3, -1)))
        print([genresRev[x] for x in np.argmax(y, axis=1)])
        
        fig, ax = plt.subplots()
        
        i = 0
        bar = plt.bar(range(10), y[i], animated=True)
        ax.set_xticklabels(list(genres.keys()))
        def updatefig(i):
            for j, b in enumerate(bar):
                b.set_height(y[i-1][j])
            return bar
        
        ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
        plt.show()



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