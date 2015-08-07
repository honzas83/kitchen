import theano
import theano.tensor as T
import numpy as np
import kitchen
import lasagne
from kitchen.text import maxpool, MaxpoolLayer, PooledNetwork, PooledVectorizer
from lasagne.objectives import categorical_crossentropy, aggregate
from lasagne.updates import sgd, apply_momentum
from lasagne.layers import InputLayer, DenseLayer, get_output, get_all_params
from lasagne.nonlinearities import sigmoid, rectify
from nose.tools import raises
from sklearn.utils.testing import assert_equal, assert_array_equal
from sklearn.utils.validation import NotFittedError
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

TEST_EPS = 1e-3

def test_maxpool():
    bounds = np.array([[0, 3], [3, 5], [5, 7], [7, 7], [7, 10]])
    X = np.random.randn(10, 3)

    W = np.random.randn(2, 3).T.astype(theano.config.floatX)
    b = np.array([2., 1.5]).astype(theano.config.floatX)
    Wp = theano.shared(W.astype(theano.config.floatX))
    bp = theano.shared(b.astype(theano.config.floatX))
    
    p0 = T.lmatrix()
    p1 = T.matrix()
    p2 = T.matrix()
    p3 = T.vector()

    L = T.mean(T.nnet.softplus(maxpool(p0, T.dot(p1, Wp)+bp)))

    cost = theano.function(inputs=[p0, p1], outputs=L, allow_input_downcast=True)
    cost_W = theano.function(inputs=[p0, p1, p2], outputs=L, givens=[(Wp, p2)], allow_input_downcast=True)
    cost_b = theano.function(inputs=[p0, p1, p3], outputs=L, givens=[(bp, p3)], allow_input_downcast=True)

    cost_grad_b = theano.function(inputs=[p0, p1], outputs=T.grad(L, bp), allow_input_downcast=True)

    cost_grad_W = theano.function(inputs=[p0, p1], outputs=T.grad(L, Wp), allow_input_downcast=True)


    grad_fin = np.zeros_like(b)
    for i in range(b.shape[0]):
        EPS = np.zeros_like(b)
        EPS[i] = 1e-3
        grad_fin[i] = (cost_b(bounds, X, b+EPS)-cost_b(bounds, X, b-EPS))/(2*EPS.sum())
    norm = (grad_fin-cost_grad_b(bounds, X)).sum()
    assert abs(norm) < TEST_EPS

    grad_fin = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            EPS = np.zeros_like(W)
            EPS[i,j] = 1e-3
            grad_fin[i,j] = (cost_W(bounds, X, W+EPS)-cost_W(bounds, X, W-EPS))/(2*EPS.sum())
    norm = (grad_fin-cost_grad_W(bounds, X)).sum() 
    assert abs(norm) < TEST_EPS

def test_maxpool_layer():
    l_in1 = InputLayer((None, 2))
    l_in2 = InputLayer((None, 20))
    l_hid = DenseLayer(l_in2, num_units=30, nonlinearity=rectify)
    l_pool = MaxpoolLayer([l_in1, l_hid])
    l_out = DenseLayer(l_pool, num_units=1, nonlinearity=sigmoid)

    bounds = theano.tensor.lmatrix('bounds')
    data = theano.tensor.matrix('data')
    targets = theano.tensor.matrix('targets')

    predictions = get_output(l_out, {l_in1: bounds, l_in2: data})
    loss = categorical_crossentropy(predictions, targets)
    loss = aggregate(loss, mode='mean')

    params = get_all_params(l_out)
    updates_sgd = sgd(loss, params, learning_rate=0.0001)

    train_function = theano.function([bounds, data, targets], updates=updates_sgd, allow_input_downcast=True)

    test_bounds = np.array([[0, 3], [3, 5], [5, 7]])
    test_X = np.random.randn(10, 20)
    test_Y = np.array([[0], [1], [0]])

    train_function(test_bounds, test_X, test_Y)


@raises(ValueError)
def test_maxpool_layer2():
    l_in1 = InputLayer((None, 2))
    l_in2 = InputLayer((None, 20))
    l_in3 = InputLayer((None, 20))
    l_hid = DenseLayer(l_in2, num_units=30, nonlinearity=rectify)
    l_pool = MaxpoolLayer([l_in1, l_hid, l_in3])

class MyPooledNetwork(PooledNetwork, kitchen.ADADelta, kitchen.BinaryCrossentropy):
    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotUniform(random_state=random_state, gain='relu')
        initb = kitchen.init.Uniform(random_state=random_state)

        i0 = lasagne.layers.InputLayer(shape=(None, X_dim[0]), input_var=T.lmatrix('bounds'))
        i1 = lasagne.layers.InputLayer(shape=(None, X_dim[1]))

        h1 = lasagne.layers.DenseLayer(i1, num_units=5, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        h2 = MaxpoolLayer([i0, h1])

        o1 = lasagne.layers.DenseLayer(h2, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return (i0, i1), o1

def test_poolednetwork():
    bounds = np.array([[0, 3], [3, 5], [5, 7]])
    X = np.random.randn(10, 3)
    y = np.array([0, 1, 0])

    clsf = MyPooledNetwork()
    clsf.fit((bounds, X), y)

    clsf.loss((bounds, X), y)

    clsf.predict((bounds, X))


def test_poolednetwork2():
    bounds = np.array([[0, 3], [3, 5], [5, 7], [7, 7], [7, 10]])
    X = np.random.randn(10, 3)
    y = np.array([0, 1, 0, 1, 0])

    clsf = MyPooledNetwork()
    clsf.fit((bounds, X), y)

    clsf.loss((bounds, X), y)

    clsf.predict((bounds, X))



def test_vectorizer():
    text = ['foo bar baz foo']

    v = PooledVectorizer(2, min_order=1)
    bounds, X = v.fit_transform(text)

    assert_array_equal(bounds, [[0, 7]])
    assert_array_equal(X, [[2, 1],
                           [3, 1],
                           [2, 3],
                           [4, 1],
                           [3, 4],
                           [2, 1],
                           [4, 2]])


def test_vectorizer2():
    text = ['foo bar baz foo']

    v = PooledVectorizer(2)
    bounds, X = v.fit_transform(text)

    assert_array_equal(bounds, [[0, 3]])
    assert_array_equal(X, [[2, 3],
                           [3, 4],
                           [4, 2]])

    assert_equal(v.get_feature_names(), [u'__padding-magic-1', u'__padding-magic-2', u'foo', u'bar', u'baz'])


def test_vectorizer3():
    text = ['foo bar baz foo', 'foo baz']

    v = PooledVectorizer(2)
    bounds, X = v.fit_transform(text)

    assert_array_equal(bounds, [[0, 3], [3, 4]])
    assert_array_equal(X, [[2, 3],
                           [3, 4],
                           [4, 2],
                           [2, 4]])


@raises(NotFittedError)
def test_vectorizer4():
    text = ['foo bar baz foo', 'foo baz']

    v = PooledVectorizer(2)
    bounds, X = v.transform(text)


def test_vectorizer5():
    text = ['foo bar baz foo', 'foo baz']

    v = PooledVectorizer(2)
    bounds, X = v.fit(text).transform(text)

    assert_array_equal(bounds, [[0, 3], [3, 4]])
    assert_array_equal(X, [[2, 3],
                           [3, 4],
                           [4, 2],
                           [2, 4]])


def test_vectorizer6():
    text = ['foo bar', 'baz']

    v = PooledVectorizer(2, sent_start='<s>')
    v.fit(text)
    bounds1, X1 = v.transform(text)

    text2 = ['foo alpha', 'foo bravo', 'foo charlie', 'foo delta']
    bounds2, X2 = v.transform(text2)

    assert X1.max() >= X2.max()


class MyPooledNetwork2(PooledNetwork, kitchen.SGDNesterovMomentum, kitchen.BinaryCrossentropy):
    def get_Xy_dim(self, X, y):
        bounds, features = X
        X_dim = (bounds.shape[1], features.shape[1], features.max()+1)
        y_dim = y.shape[1]
        return X_dim, y_dim

    def create_layers(self, X_dim, y_dim, random_state):
        initW = kitchen.init.GlorotUniform(random_state=random_state, gain='relu')
        initb = kitchen.init.Uniform(random_state=random_state)

        i0 = lasagne.layers.InputLayer(shape=(None, X_dim[0]), input_var=T.lmatrix('bounds'))
        i1 = lasagne.layers.InputLayer(shape=(None, X_dim[1]), input_var=T.lmatrix('X'))

        h1 = lasagne.layers.EmbeddingLayer(i1, input_size=X_dim[2], output_size=40, W=initW)

        h2 = lasagne.layers.DenseLayer(h1, num_units=40, nonlinearity=lasagne.nonlinearities.rectify, W=initW, b=initb)

        h3 = MaxpoolLayer([i0, h2])

        o1 = lasagne.layers.DenseLayer(h3, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid, W=initW, b=initb)

        return (i0, i1), o1

def test_pooled_net():
    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)

    newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)

    v = PooledVectorizer(3, 1)
    bounds, X = v.fit_transform(newsgroups_train.data)
    y = newsgroups_train.target

    test_bounds, test_X = v.transform(newsgroups_test.data)
    test_y = newsgroups_test.target

    clsf = MyPooledNetwork2(n_epochs=1, learning_rate=0.1)
    clsf.fit((bounds, X), y)

    pred_y = clsf.predict((test_bounds, test_X))
    print accuracy_score(test_y, pred_y)


def test_maxpool_layer_forward_pass():
    W_emb = [[0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0]]
    W_emb = np.array(W_emb)

    W_dense = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0, 0, 0, 0,-0.5, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    W_dense = np.array(W_dense, dtype=float).T

    bounds = T.lmatrix('bounds')
    X = T.lmatrix('X')

    l_in1 = InputLayer((None, 2), input_var=bounds)
    l_in2 = InputLayer((None, 2), input_var=X)
        
    h1 = lasagne.layers.EmbeddingLayer(l_in2, input_size=4, output_size=5, W=W_emb)
    h2 = lasagne.layers.FlattenLayer(h1)

    h3 = lasagne.layers.DenseLayer(h2, num_units=5, nonlinearity=rectify, W=W_dense)

    l_pool = MaxpoolLayer([l_in1, h3])

    predictions = get_output(l_pool)

    pred_func = theano.function([bounds, X], predictions, allow_input_downcast=True, on_unused_input='warn')

    test_bounds = np.array([[0, 4]])
    test_X = np.array([[0, 1], [0, 0], [1, 1], [3, 3]])

    print pred_func(test_bounds, test_X)
