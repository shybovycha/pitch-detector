# -*- coding: utf-8 -*-
import numpy as np
import pickle
from os.path import isfile
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer


class SimpleConvolutionalNetwork(FeedForwardNetwork):
    """ A network with a specific form of weight-sharing, on a single 2D layer,
    convoluting neighboring inputs (within a square). """

    def __init__(self, inputdim, insize, convSize, numFeatureMaps, **args):
        FeedForwardNetwork.__init__(self, **args)
        inlayer = LinearLayer(inputdim * insize * insize)
        self.addInputModule(inlayer)
        self._buildStructure(inputdim, insize, inlayer, convSize, numFeatureMaps)
        self.sortModules()

    def _buildStructure(self, inputdim, insize, inlayer, convSize, numFeatureMaps):
        # build layers
        outdim = insize - convSize + 1
        hlayer = TanhLayer(outdim * outdim * numFeatureMaps, name='h')
        self.addModule(hlayer)

        outlayer = SigmoidLayer(outdim * outdim, name='out')
        self.addOutputModule(outlayer)

        # build shared weights
        convConns = []
        for i in range(convSize):
            convConns.append(MotherConnection(convSize * numFeatureMaps * inputdim, name='conv' + str(i)))
        outConn = MotherConnection(numFeatureMaps)

        # establish the connections.
        for i in range(outdim):
            for j in range(outdim):
                offset = i * outdim + j
                outmod = ModuleSlice(hlayer, inSliceFrom=offset * numFeatureMaps,
                                     inSliceTo=(offset + 1) * numFeatureMaps,
                                     outSliceFrom=offset * numFeatureMaps, outSliceTo=(offset + 1) * numFeatureMaps)
                self.addConnection(
                    SharedFullConnection(outConn, outmod, outlayer, outSliceFrom=offset, outSliceTo=offset + 1))

                for k, mc in enumerate(convConns):
                    offset = insize * (i + k) + j
                    inmod = ModuleSlice(inlayer, outSliceFrom=offset * inputdim,
                                        outSliceTo=offset * inputdim + convSize * inputdim)
                    self.addConnection(SharedFullConnection(mc, inmod, outmod))


X = np.array([
    # crosses
    [0, 1, 0, 0, 1,
     0, 0, 1, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 0, 0, 0, 1],

    [0, 0, 0, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 0, 0, 0],

    [0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0],

    [0, 0, 1, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 1,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0,
     0, 0, 1, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 1,
     0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 1, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 1],

    [0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0],

    [0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     1, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 1, 0, 0],

    [0, 0, 0, 0, 0,
     1, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 0, 0, 0],

    [1, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0],

    [1, 0, 0, 0, 1,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 0, 0, 0, 1],

    # rectangle

    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],

    # triangles

    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0],

    [0, 0, 0, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 1, 0, 1,
     0, 1, 0, 0, 1,
     1, 1, 1, 1, 1],

    [1, 0, 0, 0, 0,
     1, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     1, 0, 0, 1, 0,
     1, 1, 1, 1, 1],

    [1, 1, 1, 1, 1,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0,
     1, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     1, 1, 0, 0, 0,
     1, 0, 0, 0, 0],

    [0, 0, 0, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 1, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 0, 0, 1],

    [0, 0, 0, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 1, 0, 1,
     0, 1, 1, 1, 1,
     0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0,
     0, 0, 0, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 1, 0, 1,
     0, 1, 1, 1, 1],

    [0, 0, 0, 0, 0,
     0, 0, 0, 1, 0,
     0, 0, 1, 1, 0,
     0, 1, 0, 1, 0,
     1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0,
     1, 0, 0, 0, 0,
     1, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0,
     0, 1, 0, 0, 0,
     0, 1, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 1]])

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

Y1 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]])

saved_network_filename = 'network01.pickle'

if isfile(saved_network_filename):
    print "Loading network..."

    net = pickle.load(open(saved_network_filename, 'r'))
else:
    print "Building network..."

    net = SimpleConvolutionalNetwork(1, 5, 2, 2)

    # 2 = dimensionality of the each input vector; 1 = number of output types; nb_classes = number of classes
    # ds = ClassificationDataSet(, 1, nb_classes=2)

    ds = SupervisedDataSet(25, 16)

    for i in range(len(Y)):
        ds.addSample(X[i], np.append(Y[i], np.zeros(16 - len(Y[i]))))

    trainer = BackpropTrainer(net, ds)

    print "Training network..."

    trainer.trainEpochs(1000)

    print "Saving network..."

    f = open(saved_network_filename, 'w')
    pickle.dump(net, f)
    f.close()

print "Testing:"

X1 = np.array([[0, 0, 0, 1, 0,
                0, 0, 1, 1, 0,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 0]])

print net.activate(X1[0])
