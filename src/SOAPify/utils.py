import numpy


def normalizeArray(a: numpy.ndarray):
    """Normalizes the futher axis of the given array
    
    (eg. in an array of shape (100,50,3) normalizes all the  5000 3D vectors)

    Args:
        a (numpy.ndarray): the arra to be normalized

    Returns:
        numpy.ndarray: the normalized array
    """
    norm = numpy.linalg.norm(a, axis=-1,keepdims=True)
    norm[norm == 0] = 1
    return a / norm
