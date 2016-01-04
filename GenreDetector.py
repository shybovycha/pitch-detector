# -*- coding: utf-8 -*-
import numpy as np
import pickle
from os.path import isfile
import lasagne
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import wave
import glob


class GenreDetector:
    def __init__(self):
        self.X, self.Y = [], []
        self.min_length, self.max_length = 0, 0
        self.genres = ['metal', 'dubstep', 'acoustic']

        self.dataset_x_filename = 'learn_dataset03_x.pickle'
        self.dataset_y_filename = 'learn_dataset03_y.pickle'
        self.network_filename = 'network03.pickle'

        self.net = None

    def run(self):
        self.get_network()

    def get_wave_data(self, file_path):
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

                window_size = len(data) / sample_width
                window = np.blackman(window_size)

                fragment_data = wave.struct.unpack("%dh" % window_size, data) * window
                fft_data = abs(np.fft.rfft(fragment_data)) ** 2

                wave_data = np.hstack((wave_data, fft_data[:100]))  # take only 100 frequences

        return wave_data

    def dump_to_file(self, filename, arg):
        with open(filename, 'wb') as df:
            pickle.dump(arg, df)

    def load_from_file(self, filename):
        return pickle.load(open(filename, 'rb'))

    def find_songs(self):
        self.files = []

        for genre_idx, genre in enumerate(self.genres):
            genre_path = 'songs/{0}/*.wav'.format(genre)
            self.files += [[f, genre_idx] for f in glob.iglob(genre_path)]

    def get_song_data(self, pair):
        filename, genre_idx = pair

        song_data = self.get_wave_data(filename)

        self.X.append(song_data)
        self.Y.append(genre_idx)

    def load_training_data(self):
        self.X = self.load_from_file(self.dataset_x_filename)
        self.Y = self.load_from_file(self.dataset_y_filename)

        if isinstance(self.Y[0], str):
            self.Y = [self.genres.index(y) for y in self.Y]

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.normalize_dataset()

    def normalize_dataset(self):
        lengths = [len(d) for d in self.X]
        self.min_length, self.max_length = min(lengths), max(lengths)

        threshold = 879480755.510798

        # fill smaller data chunks with zeros and
        # filter FFT data so small values are treated as zero
        self.X = np.array([np.hstack((x, np.zeros(self.max_length - len(x)))) for x in self.X])

        medians = np.median(np.array(self.X))
        print('medians:', medians)

        self.X = self.X.reshape((-1, 1, self.max_length))

        self.Y = np.array(self.Y)

    def create_training_data(self):
        input_files = self.find_songs()

        for i, pair in enumerate(input_files):
            print('>> Loading file {0} ({1} / {2})'.format(pair[0], i, len(input_files)))
            self.get_song_data(pair)

        self.normalize_dataset()

        self.dump_to_file(self.dataset_x_filename, self.X)
        self.dump_to_file(self.dataset_y_filename, self.Y)

    def load_network(self):
        self.net = self.load_from_file(self.network_filename)

    def train_network(self):
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
            input_shape=(None, 1, self.max_length),
            # layer conv2d1
            conv1d1_num_filters=25,
            conv1d1_filter_size=10,
            conv1d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv1d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=25,
            # layer conv2d2
            conv1d2_num_filters=25,
            conv1d2_filter_size=100,
            conv1d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=25,
            # dropout1
            dropout1_p=0.5,
            # dense
            dense_num_units=25,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.5,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=len(self.genres),
            # optimization method params
            update=nesterov_momentum,
            update_learning_rate=0.1,
            update_momentum=0.9,
            max_epochs=150,
            verbose=1,
        )

        print("Training network...")

        self.net = net1.fit(self.X, self.Y.astype(np.uint8))

        print("Saving network...")

        self.dump_to_file(self.network_filename, self.net)

    def get_training_data(self):
        if isfile(self.dataset_x_filename):
            self.load_training_data()
        else:
            self.create_training_data()

    def get_network(self):
        if isfile(self.network_filename):
            self.load_network()
        else:
            self.get_training_data()
            self.train_network()


if __name__ == '__main__':
    detector = GenreDetector()
    detector.run()
