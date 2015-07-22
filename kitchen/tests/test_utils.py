from kitchen.utils import iter_ngram, iter_ngram_pad
from nose.tools import assert_equal, raises


@raises(ValueError)
def test_bad_bounds():
    list(iter_ngram(range(10), 2, 3))

def test_iter_ngram():
    seq = [0, 1, 2, 3, 4]

    lst = list(iter_ngram(seq, 1))
    assert_equal(lst, [[0], [1], [2], [3], [4]])

    lst = list(iter_ngram(seq, 2))
    assert_equal(lst, [[0, 1], [1, 2], [2, 3], [3, 4]])

    lst = list(iter_ngram(seq, 3))
    assert_equal(lst, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    lst = list(iter_ngram(seq, 3, 2))
    assert_equal(lst, [[0, 1], [1, 2], [0, 1, 2], [2, 3], [1, 2, 3], [3, 4], [2, 3, 4]])

    lst = list(iter_ngram(seq, 3, 1))
    assert_equal(lst, [[0], [1], [0, 1], [2], [1, 2], [0, 1, 2], [3], [2, 3], [1, 2, 3], [4], [3, 4], [2, 3, 4]])

    lst = list(iter_ngram(seq, 3, sent_start=9))
    assert_equal(lst, [[9, 9, 0], [9, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]])

    lst = list(iter_ngram(seq, 3, sent_end=9))
    assert_equal(lst, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 9], [4, 9, 9]])

    lst = list(iter_ngram(seq, 3, sent_start=9, sent_end=9))
    assert_equal(lst, [[9, 9, 0], [9, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 9], [4, 9, 9]])

    lst = list(iter_ngram(seq, 3, 2, sent_start=9, sent_end=9))
    assert_equal(lst, [[9, 0], [9, 9, 0],
                       [0, 1], [9, 0, 1],
                       [1, 2], [0, 1, 2],
                       [2, 3], [1, 2, 3],
                       [3, 4], [2, 3, 4],
                       [4, 9], [3, 4, 9],
                       [9, 9], [4, 9, 9]])

def test_iter_ngram_pad():
    pad = [7, 8, 9]
    seq = [0, 1, 2, 3, 4]

    lst = list(iter_ngram_pad(seq, 3, 1, padding=pad))
    assert_equal(lst, [[0, 8, 9],
                       [1, 8, 9],
                       [0, 1, 9],
                       [2, 8, 9],
                       [1, 2, 9],
                       [0, 1, 2],
                       [3, 8, 9],
                       [2, 3, 9],
                       [1, 2, 3],
                       [4, 8, 9],
                       [3, 4, 9],
                       [2, 3, 4]])
    
def test_iter_ngram_pad2():
    pad = [8, 9]
    seq = [0, 1, 2, 3, 4]

    lst = list(iter_ngram_pad(seq, 3, 1, padding=pad))
    assert_equal(lst, [[0, 8, 9],
                       [1, 8, 9],
                       [0, 1, 9],
                       [2, 8, 9],
                       [1, 2, 9],
                       [0, 1, 2],
                       [3, 8, 9],
                       [2, 3, 9],
                       [1, 2, 3],
                       [4, 8, 9],
                       [3, 4, 9],
                       [2, 3, 4]])

@raises(ValueError)
def test_iter_ngram_pad3():
    pad = [9]
    seq = [0, 1, 2, 3, 4]

    lst = list(iter_ngram_pad(seq, 3, 1, padding=pad))
