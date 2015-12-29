# -*- coding: utf-8 -*-
import numpy as np
import pickle
from os.path import isfile
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import wave
import glob
from joblib import Parallel, delayed
import multiprocessing


def running_mean(x, n):
    cum_sum = np.cumsum(np.insert(x, 0, 0))
    return (cum_sum[n:] - cum_sum[:-n]) / n


def get_wave_data(file_path):
    with wave.open(file_path, 'rb') as wf:
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()

        length_in_seconds = num_frames / (frame_rate * 1.0)

        num_iterations = int(length_in_seconds * 2)
        iteration_size = int(num_frames / num_iterations)

        wave_data = []

        for fragment_num in range(0, num_iterations):
            data = wf.readframes(iteration_size)

            # window = np.blackman(len(data) / sample_width)
            # in_data = np.array(wave.struct.unpack("%dh" % (len(data) / sample_width), data)) * window
            # wave_data = np.hstack((wave_data, in_data))

            fragment_data = wave.struct.unpack("%dh" % (len(data) / sample_width), data)

            wave_data = np.hstack((wave_data, running_mean(fragment_data, n=len(fragment_data) / 2)))

    return wave_data


def dump_to_file(filename, arg):
    with open(filename, 'wb') as df:
        pickle.dump(arg, df)


def load_from_file(filename):
    return pickle.load(open(filename, 'rb'))


def find_songs():
    files = []

    for genre in ['metal', 'dubstep', 'acoustic']:
        genre_path = 'songs/{0}/*.wav'.format(genre)
        files += [[f, genre] for f in glob.iglob(genre_path)]

    return files


if __name__ == '__main__':
    X, Y = [], []


    def get_song_data(pair):
        filename, genre = pair

        song_data = get_wave_data(filename)

        X.append(song_data)
        Y.append(genre)


    num_cores = multiprocessing.cpu_count()
    input_files = find_songs()

    for i, pair in enumerate(input_files):
        print('>> Loading file {0} ({1} / {2})'.format(pair[0], i, len(input_files)))
        get_song_data(pair)

    lengths = [len(d) for d in X]
    min_length, max_length = min(lengths), max(lengths)

    print('min length:', min_length, 'max length:', max_length)

    saved_network_filename = 'network03.pickle'

    if isfile(saved_network_filename):
        print("Loading network...")

        net = pickle.load(open(saved_network_filename, 'rb'))
    else:
        print("Building network...")

        net1 = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv1d1', layers.Conv1DLayer),
                    ('maxpool1', layers.MaxPool1DLayer),
                    ('conv1d2', layers.Conv1DLayer),
                    ('maxpool2', layers.MaxPool1DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('dense', layers.DenseLayer),
                    ('dropout2', layers.DropoutLayer),
                    ('output', layers.DenseLayer),
                    ],
            # input layer
            input_shape=(None, 1, 1, max_length),
            # layer conv2d1
            conv1d1_num_filters=25,
            conv1d1_filter_size=(1, 4000),
            conv1d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv1d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(1, 4),
            # layer conv2d2
            conv1d2_num_filters=25,
            conv1d2_filter_size=(1, 4000),
            conv1d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=(2, 8),
            # dropout1
            dropout1_p=0.5,
            # dense
            dense_num_units=25,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.5,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=10,
            # optimization method params
            update=nesterov_momentum,
            update_learning_rate=0.1,
            update_momentum=0.9,
            max_epochs=500,
            verbose=0,
        )

        print("Training network...")

        net = net1.fit(X.reshape((-1, 1, 1, max_length)), Y.astype(np.uint8))

        print("Saving network...")

        f = open(saved_network_filename, 'wb')
        pickle.dump(net, f)
        f.close()
