import numpy as np


def toeplitz(A):
    r"""
    Toeplitz decomposition.

    Returns the symmetric and skew-symmetric part,

        .. math::

            S = \tfrac{1}{2} (A + A^T),\quad N = \tfrac{1}{2} (A - A^T)

    of a square matrix :math:`A`.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix to split in symmetric and skew-symmetric part.

    Returns
    -------
    S : numpy.ndarray
        Symmetric part of  `A`.
    N : numpy.ndarray
        Skew-symmetric part of  `A`.
    """
    return sym(A), skew(A)


def sym(A):
    """
    Symmetric part of a square matrix `A`.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix.

    Returns
    -------
    S : numpy.ndarray
        Symmetric part of `A`.
    """
    return 1 / 2 * (A + A.T)


def skew(A):
    """
    Skew-symmetric part of a square matrix `A`.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix.

    Returns
    -------
    N : numpy.ndarray
        Skew-symmetric part of `A`.
    """
    return 1 / 2 * (A - A.T)


def is_sym(A, rtol=1e-05, atol=1e-08):
    """
    Checks if a square matrix `A` is symmetric.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix.

    Returns
    -------
    is_sym : bool
        `True` if `A` is symmetric, else `False`.
    """
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def is_skew(A, rtol=1e-05, atol=1e-08):
    """
    Checks if a square matrix `A` is skew-symmetric.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix.

    Returns
    -------
    is_sym : bool
        `True` if `A` is skew-symmetric, else `False`.
    """
    return np.allclose(A, -A.T, rtol=rtol, atol=atol)


if __name__ == "__main__":
    n = 5
    A = np.random.random((n, n))

    A_sym, A_skew = toeplitz(A)

    assert is_sym(A_sym) and is_skew(A_skew)
