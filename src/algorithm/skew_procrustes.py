import logging

import numpy as np

from linalg.svd import svd


def skew_procrustes(X, Y, trunc_tol=1e-12, zero_tol=1e-14):
    r"""
    Solves the skew-symmetric Procrustes problem

        .. math::
            \min_{A} \|Y - AX\|_\mathrm{F} \quad s.t. \quad A = -A^T.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix.
    Y : numpy.ndarray
        Matrix.
    Returns
    -------
    A : numpy.ndarray
        The minimizer of the skew-symmetric Procrustes problem.
    e : float
        Error of the skew-symmetric Procrustes problem.
    """

    n = X.shape[0]
    U, s, V = svd(X)
    r = np.argmax(s / s[0] < trunc_tol)
    r = r if r > 0 else len(s)

    s = s[:r]

    Z_transform = U.T @ Y @ V
    Z1 = Z_transform[:r, :r]
    Z3 = Z_transform[r:, :r]

    # Assemble Phi
    [ii, jj] = np.mgrid[:r, :r]
    tmp = np.square(s[ii]) + np.square(s[jj])
    Phi = 1. / tmp

    A1 = Phi * (Z1 @ np.diag(s) - np.diag(s) @ Z1.T)

    A2 = -np.linalg.solve(np.diag(s), Z3.T)
    A4 = np.zeros((n - r, n - r))

    A = U @ np.array(np.concatenate(
        (np.hstack((A1, A2)), np.hstack((-A2.T, A4)))
    )) @ U.T

    max_A = np.amax(np.abs(A))
    indices = np.where(np.abs(A) / max_A < zero_tol)
    A[indices] = 0

    e = np.linalg.norm(A @ X - Y, 'fro')

    return A, e


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    n = 6
    M = 500
    A = np.random.random((n, n))
    A = (A - A.T) / 2

    X = np.random.random((n, M))
    r = np.linalg.matrix_rank(X)

    Y = A @ X

    X_res, res = skew_procrustes(X, Y)

    logging.info(f'Error of the skew symmetric Procrustes ||A @ X - Y||_F:\t{res:.2e}')

    assert res <= 1e-2
