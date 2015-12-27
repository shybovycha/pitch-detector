# -*- coding: utf-8 -*-
import numpy as np
import pickle
from os.path import isfile
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

X = np.array([
    # crosses
    [0, 1, 0, 0, 1, 0,
     0, 0, 1, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     1, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [1, 0, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     1, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 0, 0],

    # rectangle

    [1, 1, 1, 1, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0],

    [1, 1, 1, 1, 0, 0,
     1, 0, 0, 1, 0, 0,
     1, 0, 0, 1, 0, 0,
     1, 1, 1, 1, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1,
     0, 0, 1, 0, 0, 1,
     0, 0, 1, 0, 0, 1,
     0, 0, 1, 1, 1, 1],

    # triangles

    [0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     1, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 1, 0,
     0, 0, 0, 1, 1, 0,
     0, 0, 1, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     1, 0, 0, 1, 0, 0,
     1, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0],

    [1, 1, 1, 1, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 1, 0,
     0, 0, 0, 1, 1, 0,
     0, 0, 1, 0, 1, 0,
     0, 0, 0, 1, 1, 0,
     0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 1, 0,
     0, 0, 0, 1, 1, 0,
     0, 0, 1, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 0,
     0, 0, 0, 1, 1, 0,
     0, 0, 1, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 1, 0, 0,
     0, 1, 0, 1, 0, 0,
     1, 1, 1, 1, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0,
     1, 1, 1, 1, 0, 0,
     0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 0, 0]])

Y = np.array([[1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)
              [1, 0, 0],  # cross (X)

              [0, 1, 0],  # square
              [0, 1, 0],  # square
              [0, 1, 0],  # square
              [0, 1, 0],  # square
              [0, 1, 0],  # square

              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1],  # triangle
              [0, 0, 1]])  # triangle

# 3 - cross, 2 - circle, 1 - triangle
Y1 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

saved_network_filename = 'network02.pickle'

if isfile(saved_network_filename):
    print("Loading network...")

    net = pickle.load(open(saved_network_filename, 'rb'))
else:
    print("Building network...")

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 6, 6),
        # layer conv2d1
        conv2d1_num_filters=25,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(1, 1),
        # layer conv2d2
        conv2d2_num_filters=25,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
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

    net = net1.fit(X.reshape((-1, 1, 6, 6)), Y1.astype(np.uint8))

    print("Saving network...")

    f = open(saved_network_filename, 'wb')
    pickle.dump(net, f)
    f.close()

print("Testing:")

X1 = np.array([[0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 1, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0,
                0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0]])

print(net.predict(X1.reshape((-1, 1, 6, 6))))
