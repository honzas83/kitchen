import lasagne.layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.utils import check_random_state
import numpy as np


class DropoutLayer(lasagne.layers.DropoutLayer):
    def __init__(self, incoming, random_state=None, **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self.random_state = check_random_state(random_state)
        self._srng = RandomStreams(self.random_state.randint(1, 2147462579))


class MulLayer(lasagne.layers.Layer):
    def __init__(self, incoming, W=np.ones, **kwargs):
        super(MulLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.W = self.add_param(W, (num_inputs,), name='W')

    def get_output_for(self, input, **kwargs):
        return input * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape
