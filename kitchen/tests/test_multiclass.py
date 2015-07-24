import kitchen
import lasagne
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.testing import assert_equal


class NetSGD(kitchen.Network, kitchen.CategoricalCrossentropy, kitchen.SGD):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotUniform(random_state=random_state, gain='relu')
        initb = kitchen.init.Uniform(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))
        l02 = kitchen.layers.DropoutLayer(l0, p=0.5, random_state=random_state)

        l1 = lasagne.layers.DenseLayer(l02, num_units=128, nonlinearity=lasagne.nonlinearities.LeakyRectify(), W=initW, b=initb)
        l12 = kitchen.layers.DropoutLayer(l1, p=0.5, random_state=random_state)

        l3 = lasagne.layers.DenseLayer(l12, num_units=y_dim, nonlinearity=lasagne.nonlinearities.softmax, W=initW, b=initb)

        return l0, l3

def test_fit_sgd():
    X = np.arange(90).reshape((30, 3))
    y = np.arange(30) % 4

    net = NetSGD(random_state=42)
    net.fit(X, y)
    net.loss(X, y)
    y_pred = net.predict(X)

    assert len(net.classes_) == 4

