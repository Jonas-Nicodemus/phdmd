import numpy as np

from linalg.symmetric import is_sym, sym


def is_spsd(A):
    """
    Checks if a matrix is symmetric positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to check for symmetric positive semi-definiteness.

    Returns
    -------
    is_spsd : bool
    `True` if `A` is symmetric positive semi-definiteness, else `False`.
    """
    if is_sym(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            A += 1e-5 * np.eye(A.shape[0])
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
    else:
        return False


def project_spsd(A):
    """
    Projects a matrix on the set of symmetric positive semi-definite matrices.

    Parameters
    ----------
    A : array_like
        The matrix to project on the set of symmetric positive semi-definite matrices.

    Returns
    -------
    A_spsd : numpy.ndarray
        The projected symmetric positive semi-definite matrix.
    """
    w, U = np.linalg.eigh(sym(A))

    return U @ np.diag(w.clip(min=0)) @ U.T
