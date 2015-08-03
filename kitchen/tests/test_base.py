import kitchen
import lasagne
import numpy as np
from sklearn.base import clone
from sklearn.utils.testing import assert_equal, assert_array_almost_equal
from copy import deepcopy


class MyNetwork(kitchen.Network, kitchen.SGDNesterovMomentum, kitchen.BinaryCrossentropy):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotUniform(random_state=random_state, gain='relu')
        initb = kitchen.init.Uniform(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))
        l02 = kitchen.layers.DropoutLayer(l0, p=0.5, random_state=random_state)

        l1 = lasagne.layers.DenseLayer(l02, num_units=128, nonlinearity=lasagne.nonlinearities.LeakyRectify(), W=initW, b=initb)
        l12 = kitchen.layers.DropoutLayer(l1, p=0.5, random_state=random_state)

        l3 = lasagne.layers.DenseLayer(l12, num_units=y_dim, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return l0, l3

def test_cloning():
    net1 = MyNetwork(random_state=42)
    net2 = clone(net1)

    assert_equal(net1.get_params(), net2.get_params())

def test_fit():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net = MyNetwork(random_state=42)
    net.fit(X, y)
    net.loss(X, y)
    y_pred = net.predict(X)

def test_fit_copy():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net1 = MyNetwork(random_state=42)
    net1.fit(X, y)
    net1.loss(X, y)

    net2 = deepcopy(net1)
    y_pred = net2.predict(X)

    assert_equal(net1.get_params(), net2.get_params())

def test_fit_batch_size_1():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[0], [1]])

    net = MyNetwork(random_state=42, batch_size=1)
    net.fit(X, y)

def test_fit_epoch_callback():
    X = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
    y = np.array([0, 1, 0])

    num_callback = [0]

    num_batch_callbacks = [0]

    def callback(stats):
        num_callback[0] += 1

    def batch_callback(stats):
        num_batch_callbacks[0] += 1

    N_EPOCHS = 5

    net = MyNetwork(random_state=42, batch_size=1, n_epochs=N_EPOCHS, epoch_callback=callback, batch_callback=batch_callback)
    net.fit(X, y)

    assert num_callback[0] == N_EPOCHS
    assert num_batch_callbacks[0] == N_EPOCHS * X.shape[0]

def test_fit_epoch_callback_stop():
    X = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
    y = np.array([0, 1, 0])

    num_callback = [0]

    N_EPOCHS = 5
    N_STOP = 3

    def callback(stats):
        num_callback[0] += 1
        if num_callback[0] >= N_STOP:
            raise StopIteration

    net = MyNetwork(random_state=42, batch_size=1, n_epochs=N_EPOCHS, epoch_callback=callback)
    net.fit(X, y)

    assert num_callback[0] == N_STOP

def test_fit_multi_binary():
    X = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 5], [1, 4, 2]])
    y = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1]])

    net = MyNetwork(random_state=42)
    net.fit(X, y)
    net.loss(X, y)
    y_pred = net.predict_proba(X)

    assert_array_almost_equal(y_pred.sum(2), np.ones(y_pred.shape[:2]))

