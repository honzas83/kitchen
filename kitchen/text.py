'''Text and lattice processing layers
'''
import theano
import numpy as np
import six
import lasagne
from operator import itemgetter
from collections import defaultdict
from . import Network
from .utils import iter_ngram_pad
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin


def maxpool_grad_perform(bounds, input, maximums, out_mtx):
    z = np.zeros_like(input)

    i0 = np.arange(z.shape[1])

    for idx in xrange(maximums.shape[0]):
        o_val = out_mtx[idx]
        m_row = maximums[idx]
        z[m_row, i0] = o_val
    return z


def maxpool_perform(bounds, z):
    a = np.zeros((bounds.shape[0], z.shape[1]), dtype=z.dtype)
    m = np.zeros((bounds.shape[0], z.shape[1]), dtype=bounds.dtype)

    i0 = np.arange(z.shape[1])

    for m_idx, (b0, b1) in enumerate(bounds):
        m[m_idx] = np.argmax(z[b0:b1], 0)+b0
        a[m_idx] = z[m[m_idx], i0]
    return a, m


class MaxPoolGrad(theano.gof.Op):
    def make_node(self, ridxs, z, maximums, out_mtx):
        return theano.Apply(self, [ridxs, z, maximums, out_mtx], [z.type()])

    def perform(self, node, inputs, output_storage):
        ridxs, z, maximums, out_mtx = inputs
        output_storage[0][0] = maxpool_grad_perform(ridxs, z, maximums, out_mtx)

    def infer_shape(self, node, inputs):
        ridxs, z, maximums, out_mtx = inputs
        return [z]

maxpool_grad = MaxPoolGrad()


class MaxPool(theano.gof.Op):
    default_output = 0

    def make_node(self, bounds, z):
        node = theano.Apply(self, [bounds, z], [z.type.make_variable(), bounds.type.make_variable()])
        self.maximums = node.outputs[1]
        return node

    def perform(self, node, inputs, output_storage):
        bounds, z = inputs
        output_storage[0][0], output_storage[1][0] = maxpool_perform(bounds, z)

    def infer_shape(self, node, inputs):
        bounds, z = inputs
        return [(bounds[0], z[1]), (bounds[0], z[1])]

    def grad(self, inputs, output_gradients):
        bounds, z = inputs
        out_mtx, _ = output_gradients
        return [bounds*0.,
                maxpool_grad(bounds, z, self.maximums, out_mtx),
                ]

maxpool = MaxPool()


class MaxpoolLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        if len(incomings) != 2:
            raise ValueError("incomings must be of length 2!")
        super(MaxpoolLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        ret = input_shapes[0][0], input_shapes[1][1]
        return ret

    def get_output_for(self, inputs, **kwargs):
        return maxpool(inputs[0], inputs[1])


class PooledNetwork(Network):
    def iter_batches(self, X, y, shuffle=False):
        bounds, features = X

        batch_num = bounds.shape[0] // self.batch_size
        if batch_num == 0 or bounds.shape[0] % self.batch_size != 0:
            batch_num += 1

        idxs = range(batch_num)
        if shuffle:
            self.random_state_.shuffle(idxs)

        for idx in idxs:
            slc = slice(idx*self.batch_size, (idx+1)*self.batch_size)

            bounds_slc = bounds[slc]
            bmin = bounds_slc.min()
            bmax = bounds_slc.max()

            features_slc = features[bmin:bmax]

            if y is None:
                yield bounds_slc-bmin, features_slc,
            else:
                yield bounds_slc-bmin, features_slc, y[slc]

    def get_Xy_dim(self, X, y):
        bounds, features = X
        X_dim = (bounds.shape[1], features.shape[1])
        y_dim = y.shape[1]
        return X_dim, y_dim


class PooledVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, max_order, min_order=None, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 analyzer='word',
                 sent_start=None, sent_end=None,
                 vocabulary=None, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_order = max_order
        self.min_order = min_order
        self.vocabulary = vocabulary
        self.dtype = dtype
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.ngram_range = (1, 1)

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()

        padding = []
        for i in range(1, self.max_order+1):
            key = '__padding-magic-%d' % i
            vocabulary[key]  # introduce the unique key into the vocabulary
            padding.append(key)

        X = []
        bounds = []
        idx = 0
        for doc in raw_documents:
            features = list(analyze(doc))
            vdoc = []
            idx0 = idx
            for ngram in iter_ngram_pad(features, self.max_order, self.min_order, self.sent_start, self.sent_end, padding=padding):
                vector = []
                for k, i in enumerate(ngram, 1):
                    try:
                        vector.append(vocabulary[i])
                    except KeyError:
                        vector.append(vocabulary['__padding-magic-%d' % k])
                vdoc.append(vector)
                idx += 1
            idx1 = idx
            if idx0 != idx1:
                bounds.append(idx0)
                bounds.append(idx1)
                vdoc = np.array(vdoc, dtype=self.dtype)
                X.append(vdoc)

        X = np.vstack(X)
        bounds = np.array(bounds, dtype=self.dtype).reshape((-1, 2))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        return vocabulary, X, bounds

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        self._validate_vocabulary()

        vocabulary, X, bounds = self._count_vocab(raw_documents, self.fixed_vocabulary_)

        if not self.fixed_vocabulary_:
            self.vocabulary_ = vocabulary

        return bounds, X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X, bounds = self._count_vocab(raw_documents, fixed_vocab=True)
        return bounds, X

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        self._check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]
