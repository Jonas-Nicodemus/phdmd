import numpy as np
from pymor.models.iosys import LTIModel, PHLTIModel


def unstack(AA, n, no_feedtrough=False):
    """
    Unstack a matrix into blocks.

    Parameters
    ----------
    AA : numpy.ndarray
        Matrix to unstack.
    n : int
        Number of rows/columns in the first block.
    no_feedtrough
        If `True`, the feedtrough block is set to zero.

    Returns
    -------
    A : numpy.ndarray
        First block.
    B : numpy.ndarray
        Second block.
    C : numpy.ndarray
        Third block.
    D : numpy.ndarray
        Fourth block.
    """
    A = AA[:n, :n]
    B = AA[:n, n:]
    C = AA[n:, :n]
    D = AA[n:, n:]

    if no_feedtrough:
        D = np.zeros_like(D)

    return A, B, C, D


def to_phlti(JJ, RR, n, E=None, no_feedtrough=False, name=None):
    """
    Creates a PHLTIModel from structure matrix and dissipation matrix.

    Parameters
    ----------
    JJ : numpy.ndarray
        Structure matrix.
    RR : numpy.ndarray
        Dissipation matrix.
    n : int
        State dimension.
    E : numpy.ndarray
        Hamitlonian matrix.
    no_feedtrough : bool
        If `True`, the feedtrough block is set to zero.
    name : str
        Name of the model.

    Returns
    -------
    model : pymor.models.iosys.PHLTIModel
        The PHLTIModel.
    """
    J, G, _, N = unstack(JJ, n, no_feedtrough)
    R, P, _, S = unstack(RR, n, no_feedtrough)

    return PHLTIModel.from_matrices(J, R, G, P, S, N, E, name=name)


def to_lti(AA, n, E=None, sampling_time=0, no_feedtrough=False, name=None):
    """
    Creates a LTIModel from block matrix.

    Parameters
    ----------
    AA : numpy.ndarray
        Block matrix.
    n : int
        State dimension.
    E : numpy.ndarray
        E matrix.
    sampling_time : float
        Sampling time.
    no_feedtrough : bool
        If `True`, the feedtrough block is set to zero.
    name : str
        Name of the model.

    Returns
    -------
    model : pymor.models.iosys.LTIModel
        The LTIModel.
    """
    A, B, C, D = unstack(AA, n, no_feedtrough)

    return LTIModel.from_matrices(A, B, C, D, E=E, sampling_time=sampling_time, name=name)
