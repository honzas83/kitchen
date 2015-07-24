import warnings
warnings.filterwarnings('ignore', '.*scipy.linalg.blas.fblas.*')
warnings.filterwarnings('ignore', '.*Glorot.*')
warnings.filterwarnings('ignore', '.*topo.*')

import lasagne
import theano.tensor as T
import theano
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime as dt
import itertools
import inspect

from . import init, layers

__all__ = ['init', 'layers', 'text', 'utils']


class Network(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None, batch_size=128, n_epochs=5,
                 epoch_callback=None, batch_callback=None, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_callback = epoch_callback
        self.batch_callback = batch_callback

    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        mro_args = []
        for cls in self.__mro__:
            try:
                # fetch the constructor or the original constructor before
                # deprecation wrapping if any
                init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

                # introspect the constructor arguments to find the model parameters
                # to represent
                args, varargs, kw, default = inspect.getargspec(init)
                # Remove 'self'
                # XXX: This is going to fail if the init is a staticmethod, but
                # who would do this?
                args.pop(0)
            except TypeError:
                # No explicit __init__
                args = []
            mro_args.extend(args)
        mro_args.sort()
        return mro_args

    def _get_inputs(self):
        if isinstance(self.input_layer_, (list, tuple)):
            iter = self.input_layer_
        else:
            iter = [self.input_layer_]

        ret_list = []
        ret_dict = {}
        for i in iter:
            ret_list.append(i.input_var)
            ret_dict[i] = i.input_var
        return ret_list, ret_dict

    def iter_batches(self, X, y, shuffle=False):
        batch_num = X.shape[0] // self.batch_size
        if batch_num == 0 or X.shape[0] % self.batch_size != 0:
            batch_num += 1

        idxs = range(batch_num)
        if shuffle:
            self.random_state_.shuffle(idxs)

        for batch_num, idx in enumerate(idxs, 1):
            slc = slice(idx*self.batch_size, (idx+1)*self.batch_size)

            if y is None:
                yield X[slc],
            else:
                yield X[slc], y[slc]

    def get_Xy_dim(self, X, y):
        X_dim = X.shape[1]
        y_dim = y.shape[1]
        return X_dim, y_dim

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)

        X, y = self._transform_Xy(X, y, fit=True)

        X_dim, y_dim = self.get_Xy_dim(X, y)
        self.input_layer_, self.output_layer_ = self.create_layers(X_dim, y_dim, self.random_state_)

        num_samples = y_dim

        tX, var_X = self._get_inputs()

        ty = T.matrix('y')

        train_loss, test_loss = self.create_loss(var_X, ty)

        try:
            train_loss = train_loss + 1./float(num_samples) * self.create_regularization(self.output_layer_)
        except AttributeError:
            pass

        all_params = lasagne.layers.get_all_params(self.output_layer_, trainable=True)

        updates = self.create_updates(train_loss, all_params)

        self._train_func = theano.function(tX+[ty], train_loss, updates=updates, allow_input_downcast=True)
        self._test_func = theano.function(tX+[ty], test_loss, allow_input_downcast=True)
        self._predict_func = theano.function(tX, lasagne.layers.get_output(self.output_layer_, var_X, deterministic=True), allow_input_downcast=True)

        t0 = dt.datetime.now()
        for epoch in itertools.count(1):
            t1 = dt.datetime.now()
            batch_train_losses = []

            for batch_num, args in enumerate(self.iter_batches(X, y, shuffle=True), 1):
                batch_train_loss = self._train_func(*args)
                batch_train_losses.append(batch_train_loss)

                stats = dict(
                    batch_train_loss=batch_train_loss,
                    batch_num=batch_num,
                )

                if self.batch_callback:
                    self.batch_callback(stats)

            avg_train_loss = np.mean(batch_train_losses)

            t2 = dt.datetime.now()

            stats = dict(
                avg_train_loss=avg_train_loss,
                epoch=epoch,
                t_start=t0,
                t_epoch=t2-t1,
                t_total=t2-t0,
            )

            if self.epoch_callback:
                try:
                    self.epoch_callback(stats)
                except StopIteration:
                    break

            if self.n_epochs is not None and epoch >= self.n_epochs:
                break

    def loss(self, X, y):
        X, y = self._transform_Xy(X, y)

        total_loss = 0.
        total_examples = 0
        for batch_num, args in enumerate(self.iter_batches(X, y), 1):
            num_examples = args[-1].shape[0]
            total_loss += self._test_func(*args) * num_examples
            total_examples += num_examples
        return total_loss / total_examples

    def _predict_raw(self, X):
        predictions = []
        for batch_num, args in enumerate(self.iter_batches(X, None), 1):
            batch_predict = self._predict_func(*args)
            predictions.append(batch_predict)

        return np.vstack(predictions)


class BinaryCrossentropy(object):
    def _transform_Xy(self, X, y, fit=False):
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        return X, y

    def create_loss(self, input_var, output_var):
        pred_train = lasagne.layers.get_output(self.output_layer_, input_var)
        loss_train = lasagne.objectives.binary_crossentropy(pred_train, output_var)
        loss_train = lasagne.objectives.aggregate(loss_train, mode='mean')

        pred_test = lasagne.layers.get_output(self.output_layer_, input_var, deterministic=True)
        loss_test = lasagne.objectives.binary_crossentropy(pred_test, output_var)
        loss_test = lasagne.objectives.aggregate(loss_test, mode='mean')

        return loss_train, loss_test

    def predict(self, X):
        ret = self._predict_raw(X)
        return (ret > 0.5).astype(int)

    def predict_proba(self, X):
        ret = self._predict_raw(X)
        if len(ret.shape) == 1 or ret.shape[1] == 1:
            return np.hstack([1.-ret, ret])
        else:
            ret = ret[:, :, np.newaxis]
            X0 = 1.-ret
            return np.dstack([X0, ret])


class CategoricalCrossentropy(object):
    def _transform_Xy(self, X, y, fit=False):
        if fit:
            self.encoder_ = OneHotEncoder(sparse=False).fit(y[:, np.newaxis])
            self.classes_ = self.encoder_.active_features_

        y = self.encoder_.transform(y[:, np.newaxis])

        return X, y

    def create_loss(self, input_var, output_var):
        pred_train = lasagne.layers.get_output(self.output_layer_, input_var)
        loss_train = lasagne.objectives.categorical_crossentropy(pred_train, output_var)
        loss_train = lasagne.objectives.aggregate(loss_train, mode='mean')

        pred_test = lasagne.layers.get_output(self.output_layer_, input_var, deterministic=True)
        loss_test = lasagne.objectives.categorical_crossentropy(pred_test, output_var)
        loss_test = lasagne.objectives.aggregate(loss_test, mode='mean')

        return loss_train, loss_test

    def predict(self, X):
        proba = self._predict_raw(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        return self._predict_raw(X)


class ModifiedHuber(BinaryCrossentropy):
    def _loss_impl(self, predictions, target):
        p = (predictions*2)-1
        y = (target*2)-1
        z = p * y

        return T.switch(T.ge(z, 1.0), 0., T.switch(T.ge(z, -1.0), (1.0 - z)**2, -4.0 * y))

    def create_loss(self, input_var, output_var):
        pred_train = lasagne.layers.get_output(self.output_layer_, input_var)
        loss_train = self._loss_impl(pred_train, output_var)
        loss_train = lasagne.objectives.aggregate(loss_train, mode='mean')

        pred_test = lasagne.layers.get_output(self.output_layer_, input_var, deterministic=True)
        loss_test = self._loss_impl(pred_test, output_var)
        loss_test = lasagne.objectives.aggregate(loss_test, mode='mean')

        return loss_train, loss_test


class RMSProp(object):
    def __init__(self, learning_rate=1., rho=0.9, **kwargs):
        super(RMSProp, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho

    def create_updates(self, loss, params):
        updates = lasagne.updates.rmsprop(loss, params, self.learning_rate, self.rho)
        return updates


class ADADelta(object):
    def __init__(self, learning_rate=1., **kwargs):
        super(ADADelta, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def create_updates(self, loss, params):
        updates = lasagne.updates.adadelta(loss, params, self.learning_rate)
        return updates


class ADAM(object):
    def __init__(self, learning_rate=1., **kwargs):
        super(ADAM, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def create_updates(self, loss, params):
        updates = lasagne.updates.adam(loss, params, self.learning_rate)
        return updates


class SGD(object):
    def __init__(self, learning_rate=1., **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def create_updates(self, loss, params):
        updates = lasagne.updates.sgd(loss, params, self.learning_rate)
        return updates


class SGDNesterovMomentum(object):
    def __init__(self, learning_rate=1., momentum=0.9, **kwargs):
        super(SGDNesterovMomentum, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def create_updates(self, loss, params):
        updates = lasagne.updates.nesterov_momentum(loss, params, self.learning_rate, self.momentum)
        return updates


class L2Regularization(object):
    def __init__(self, alpha=0.01, **kwargs):
        super(L2Regularization, self).__init__(**kwargs)
        self.alpha = alpha

    def create_regularization(self, output_layer):
        return self.alpha * lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
