import numpy as np
import lasagne.init
from lasagne.utils import floatX
from sklearn.utils import check_random_state


class Normal(lasagne.init.Normal):
    def __init__(self, random_state=None, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.random_state = check_random_state(random_state)

    def sample(self, shape):
        return floatX(self.random_state.normal(self.mean, self.std, size=shape))


class Uniform(lasagne.init.Uniform):
    def __init__(self, random_state=None, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)
        self.random_state = check_random_state(random_state)

    def sample(self, shape):
        return floatX(self.random_state.uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Glorot(lasagne.init.Initializer):
    def __init__(self, initializer, random_state=None, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.random_state = random_state
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(random_state=self.random_state, std=std).sample(shape)


class GlorotNormal(Glorot):
    def __init__(self, random_state=None, gain=1.0, c01b=False):
        super(GlorotNormal, self).__init__(Normal, random_state, gain, c01b)


class GlorotUniform(Glorot):
    def __init__(self, random_state=None, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, random_state, gain, c01b)
