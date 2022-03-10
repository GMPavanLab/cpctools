import SOAPify
import numpy
from numpy.testing import assert_array_equal


def test_norm1D():
    a = numpy.array([2.0, 0.0])
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(na, numpy.array([1.0, 0.0]))


def test_norm2D():
    a = numpy.array([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(na, numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_norm3D():
    a = numpy.array([[[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]],[[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]])
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(na, numpy.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]))
    