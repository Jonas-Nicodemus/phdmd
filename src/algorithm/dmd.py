import logging

import numpy as np


def iodmd(X, Y, U, X1=None, E=None):
    """
    Input Output Dynamic Mode Decomposition.

    Parameters
    ----------
    X : numpy.ndarray
        Sequence of states.
    Y : numpy.ndarray
        Sequence of outputs.
    U : numpy.ndarray
        Sequence of inputs.
    X1 : numpy.ndarray, optional
        Shifted sequence of states. If not given the shifted matrix is obtained from `X`.
    E : numpy.ndarray, optional
        E matrix of the LTI system. If not given this is assumed to be the identity.

    Returns
    -------
    A : numpy.ndarray
        The identified state matrix.
    e : float
        The relative error of the DMD problem.
    """
    if X1 is None:
        X1 = X[:, 1:]
        X0 = X[:, :-1]
        U = U[:, :-1]
        Y = Y[:, :-1]
    else:
        X0 = X

    if E is None:
        E = np.eye(X.shape[0])

    T = np.concatenate((X0, U))
    Z = np.concatenate((E @ X1, Y))
    logging.info('Perform ioDMD')

    # Solve ioDMD
    A = Z @ np.linalg.pinv(T, rcond=1e-12)

    e = np.linalg.norm(Z - A @ T) / np.linalg.norm(Z)
    logging.info('ioDMD Result')
    logging.info(f'|Z - A T|_F / |Z|_F = {e:.2e}')

    return A, e


def operator_inference(X, Y, U, delta_t, E=None):
    """
    Operator Inference.

    Parameters
    ----------
    X : numpy.ndarray
        Sequence of states.
    Y : numpy.ndarray
        Sequence of outputs.
    U : numpy.ndarray
        Sequence of inputs.
    delta_t : float, optional
        Shifted sequence of states. If not given the shifted matrix is obtained from `X`.
    E : numpy.ndarray, optional
        E matrix of the LTI system. If not given this is assumed to be the identity.

    Returns
    -------
    A : numpy.ndarray
        The identified state matrix.
    e : float
        The relative error of the OI problem.
    """

    dXdt = (X[:, 1:] - X[:, :-1]) / delta_t
    X = 1 / 2 * (X[:, 1:] + X[:, :-1])
    U = 1 / 2 * (U[:, 1:] + U[:, :-1])
    Y = 1 / 2 * (Y[:, 1:] + Y[:, :-1])

    return iodmd(X, Y, U, dXdt, E)
