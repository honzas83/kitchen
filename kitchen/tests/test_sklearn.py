import kitchen
import lasagne
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.testing import assert_equal, assert_array_almost_equal
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits

class MyNetwork(kitchen.BinaryCrossentropy, kitchen.Network, kitchen.SGDNesterovMomentum):
    def __init__(self, n_units=10, **kwargs):
        super(MyNetwork, self).__init__(**kwargs)
        self.n_units = n_units

    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotUniform(random_state=random_state)
        initb = kitchen.init.Uniform(random_state=random_state)

        l0 = lasagne.layers.InputLayer(shape=(None, X_dim))

        l1 = lasagne.layers.DenseLayer(l0, num_units=self.n_units, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        l2 = lasagne.layers.DenseLayer(l1, num_units=self.n_units, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        l3 = lasagne.layers.DenseLayer(l2, num_units=self.n_units, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        l4 = lasagne.layers.DenseLayer(l3, num_units=y_dim, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return l0, l4

def test_fit():
    data = load_digits()
    X = data.data
    y = (data.target == 0)

    net = MyNetwork(random_state=42, n_epochs=10)

    grid = {
        'n_units': [100, 500],
        'learning_rate': [0.1, 0.01],
        'batch_size': [128],
    }

    cv = GridSearchCV(net, grid, scoring='accuracy')
    cv.fit(X, y)

    print cv.grid_scores_
    print cv.best_score_
    print cv.best_estimator_

    y_prob = cv.predict_proba(X)
    assert_array_almost_equal(y_prob.sum(1), np.ones(y_prob.shape[0]))


