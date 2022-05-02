import numpy


def normalizeArray(a: numpy.ndarray) -> numpy.ndarray:
    """Normalizes the futher axis of the given array

    (eg. in an array of shape (100,50,3) normalizes all the  5000 3D vectors)

    Args:
        a (numpy.ndarray): the arra to be normalized

    Returns:
        numpy.ndarray: the normalized array
    """
    norm = numpy.linalg.norm(a, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return a / norm


def fillsingleSOAPVectorFromdscribe(
    soapFromdscribe: numpy.ndarray, l_max: int = 8, n_max: int = 8
) -> numpy.ndarray:
    """Given the resul of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray): the result of the SOAP calculation from the dscribe utility
        l_max (int, optional): the l_max specified in the calculation. Defaults to 8.
        n_max (int, optional): the n_max specified in the calculation. Defaults to 8.

    Returns:
        numpy.ndarray: The full soap spectrum, with the symmetric part sored explicitly
    """
    completeData = numpy.zeros((l_max + 1, n_max, n_max), dtype=soapFromdscribe.dtype)
    limited = 0
    for l in range(l_max + 1):
        for n in range(n_max):
            for np in range(n, n_max):
                completeData[l, n, np] = soapFromdscribe[limited]
                completeData[l, np, n] = soapFromdscribe[limited]
                limited += 1

    return completeData.reshape((-1))


# TODO: adapt this to multy-atomic soap datasets
def fillSOAPVectorFromdscribe(
    soapFromdscribe: numpy.ndarray, l_max: int, n_max: int
) -> numpy.ndarray:
    """Given the resul of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray): the result of the SOAP calculation from the dscribe utility
        l_max (int, optional): the l_max specified in the calculation. Defaults to 8.
        n_max (int, optional): the n_max specified in the calculation. Defaults to 8.

    Returns:
        numpy.ndarray: The full soap spectrum, with the symmetric part sored explicitly
    """
    limitedSOAPdim = int(((l_max + 1) * (n_max + 1) * n_max) / 2)
    if soapFromdscribe.shape[-1] != limitedSOAPdim:
        raise Exception(
            "fillSOAPVectorFromdscribe: the given soap vector do not have the expected dimensions"
        )
    if len(soapFromdscribe.shape) == 1:
        return fillsingleSOAPVectorFromdscribe(soapFromdscribe, l_max, n_max)
    fullSOAPdim = (l_max + 1) * n_max * n_max
    retShape = list(soapFromdscribe.shape)
    retShape[-1] = fullSOAPdim
    retdata = numpy.empty((retShape), dtype=soapFromdscribe.dtype)
    if len(retShape) == 2:
        for i in range(soapFromdscribe.shape[0]):
            retdata[i] = fillsingleSOAPVectorFromdscribe(
                soapFromdscribe[i], l_max, n_max
            )
    elif len(retShape) == 3:
        for i in range(soapFromdscribe.shape[0]):
            for j in range(soapFromdscribe.shape[1]):
                retdata[i, j] = fillsingleSOAPVectorFromdscribe(
                    soapFromdscribe[i, j], l_max, n_max
                )
    else:
        raise Exception("fillSOAPVectorFromdscribe: cannot convert array with shape >3")

    return retdata
