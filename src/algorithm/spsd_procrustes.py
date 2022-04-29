import logging

import matplotlib.pyplot as plt
import numpy as np

from linalg.definiteness import project_spsd


def spsd_procrustes(X, Y, A0=None, max_iter=1000):
    """
    Solves the symmetric positive semi-definite Procrustes problem via the fast-gradient method

        .. math::
            \min_{A} \| Y - A X \|_\mathrm{F} \quad s.t. \quad A\in\mathbb{S}^{n}_{\succeq}.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix.
    Y : numpy.ndarray
        Matrix.
    A0 : numpy.ndarraym, optional
        Initial matrix for `A`. Default `None`.
    max_iter : int, optional
        Maximum number of iterations. Default 1000.

    Returns
    -------
    A : numpy.ndarray
        Symmetric positive-definite matrix that (most) closely maps `X` to `Y`.
    e : float
        Remaining error of the symmetric positive-definite Procrustes problem.
    """
    if A0 is None:
        A0, _ = init_spsd_procrustes(X, Y)

    A = A0

    # Precomputations
    XXt = X @ X.T
    x, _ = np.linalg.eigh(XXt)
    Lx = max(x)  # Lipschitz constant
    mux = min(x)
    qx = mux / Lx
    YXt = Y @ X.T

    # Parameters and initalization
    alpha0 = 0.1  # Parameter of the FGM in (0,1) -it can be tuned.

    beta = np.zeros(max_iter)
    alpha = np.zeros(max_iter + 1)
    e = np.zeros(max_iter + 1)

    B = A
    alpha[0] = alpha0
    e[0] = np.linalg.norm(A @ X - Y, 'fro')

    for i in range(max_iter):
        # Previous iterate
        Ap = A

        # Projected gradient step from Y
        A = project_spsd(B - (B @ XXt - YXt) / Lx)

        # FGM Coefficients
        alpha[i + 1] = (np.sqrt((alpha[i] ** 2 - qx) ** 2 + 4 * alpha[i] ** 2) + (qx - alpha[i] ** 2)) / 2
        beta[i] = alpha[i] * (1 - alpha[i]) / (alpha[i] ** 2 + alpha[i + 1])

        # Linear combination of iterates
        B = A + beta[i] * (A - Ap)

        e[i + 1] = np.linalg.norm(A @ X - Y, 'fro')

    return A, e


def init_spsd_procrustes(X, Y):
    """
    Returns an initialization of the symmetric positive semi-definite Procrustes problem.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix.
    Y : numpy.ndarray
        Matrix.

    Returns
    -------
    A0 : numpy.ndarray
        Initial matrix for `A` for the Symmetric positive-definite Procrustes problem.
    e0 : float
        Error for the initial matrix `A` of the symmetric positive-definite Procrustes problem.
    """
    n = Y.shape[0]

    A0 = np.zeros((n, n))
    for i in range(n):
        A0[i, i] = max(0, X[i, :] @ Y[i, :].T) / np.linalg.norm(X[i, :]) ** 2 + 1e-6

    e0 = np.linalg.norm(A0 @ X - Y, 'fro')

    return A0, e0


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    n = 100
    m = 1200

    A = np.random.randn(n, n)
    A = 1 / 2 * (A + A.T)
    A = A + n * np.eye(n)

    X = np.random.randn(n, m)
    Y = A @ X

    A0 = None

    A_proc, e = spsd_procrustes(X, Y, A0=A0, max_iter=200)

    e_rel = e / np.linalg.norm(Y, 'fro')
    plt.semilogy(e_rel)
    plt.show()

    logging.info(f'||Y - AX||_F = {e[-1]:.2e}')
    logging.info(f'||Y - AX||_F / ||Y||_F = {e_rel[-1]:.2e}')
