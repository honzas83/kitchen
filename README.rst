kitchen
=======

Kitchen is a Python library for artificial neural network with interface compatible with `sklearn
(scikit-learn) <http://scikit-learn.org>`_.  It is a glue code between `Lasagne
<http://lasagne.readthedocs.org/en/latest/>`_ and `sklearn <scikit-learn.org>`_. The library is
powered with the `Theano <http://deeplearning.net/software/theano/>`_ allowing fast CPU/GPGPU
computing.

The library is attempts to comply with the `sklearn coding guidelines
<http://scikit-learn.org/stable/developers/#coding-guidelines>`_, especially:

* Simple sklearn predictor API (instantiate, ``fit()``, ``predict()``)
* Handling of random numbers (consistent results between different runs of your code)
* Support for sklearn ``get_params()`` and ``set_params()``
* Pipeline compatibility
* Pickle compatibility

Installing
----------

Dependencies:

* `numpy <https://github.com/numpy/numpy>`_
* `Lasagne <https://github.com/Lasagne/Lasagne>`_
* `Theano <https://github.com/Theano/Theano>`_
* `sklearn <https://github.com/scikit-learn/scikit-learn>`_

To install using Python PIP (including all dependencies), use::

    pip install git+https://github.com/honzas83/kitchen --process-dependency-links

Examples
--------

* `Multiclass classification of text using TF-IDF and kitchen (20 newsgroups dataset) <examples/twenty_newsgroups.ipynb>`_
