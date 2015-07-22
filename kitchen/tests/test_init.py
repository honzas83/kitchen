import kitchen
import lasagne
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.testing import assert_equal
from nose.tools import raises


class NetInitNormal(kitchen.Network, kitchen.ADADelta, kitchen.BinaryCrossentropy):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.Normal(random_state=random_state)
        initb = kitchen.init.Normal(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))
        l02 = kitchen.layers.DropoutLayer(l0, p=0.5, random_state=random_state)

        l1 = lasagne.layers.DenseLayer(l02, num_units=128, nonlinearity=lasagne.nonlinearities.LeakyRectify(), W=initW, b=initb)
        l12 = kitchen.layers.DropoutLayer(l1, p=0.5, random_state=random_state)

        l3 = lasagne.layers.DenseLayer(l12, num_units=y_dim, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return l0, l3

def test_fit_normal():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net = NetInitNormal(random_state=42)
    net.fit(X, y)
    net.loss(X, y)
    y_pred = net.predict(X)

class NetGlorotFail(kitchen.Network, kitchen.ADADelta, kitchen.BinaryCrossentropy):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotNormal(random_state=random_state)
        initb = kitchen.init.GlorotNormal(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))
        l02 = kitchen.layers.DropoutLayer(l0, p=0.5, random_state=random_state)

        l1 = lasagne.layers.DenseLayer(l02, num_units=128, nonlinearity=lasagne.nonlinearities.LeakyRectify(), W=initW, b=initb)
        l12 = kitchen.layers.DropoutLayer(l1, p=0.5, random_state=random_state)

        l3 = lasagne.layers.DenseLayer(l12, num_units=y_dim, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return l0, l3

@raises(RuntimeError)
def test_fit_glorot_fail():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net = NetGlorotFail(random_state=42)
    net.fit(X, y)

class NetGlorotFail2(kitchen.Network, kitchen.ADADelta, kitchen.BinaryCrossentropy):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotNormal(random_state=random_state, c01b=True)
        initb = kitchen.init.Uniform(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))
        l02 = kitchen.layers.DropoutLayer(l0, p=0.5, random_state=random_state)

        l1 = lasagne.layers.DenseLayer(l02, num_units=128, nonlinearity=lasagne.nonlinearities.LeakyRectify(), W=initW, b=initb)
        l12 = kitchen.layers.DropoutLayer(l1, p=0.5, random_state=random_state)

        l3 = lasagne.layers.DenseLayer(l12, num_units=y_dim, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return l0, l3

@raises(RuntimeError)
def test_fit_glorot_fail2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net = NetGlorotFail2(random_state=42)
    net.fit(X, y)
