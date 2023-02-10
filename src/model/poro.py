import os
import pkg_resources

import numpy as np

from scipy.io import loadmat


def poro(n=980):
    """
    Returns a port-Hamiltonian model of linear poroelasticity in a
    bounded Lipschitz domain as described in :cite:`AltMU21`.

    Parameters
    ----------
    n : int, optional
        System dimension (can only be either: 320, 980, or 1805). Default = 980.

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix.
    J : numpy.ndarray
        Skew-symmetric matrix.
    R : numpy.ndarray
        Dissipative matrix.
    G : numpy.ndarray
        Input matrix.
    P : numpy.ndarray
        Input matrix.
    S : numpy.ndarray
        Symmetric part of feed trough matrix.
    N : numpy.ndarray
        Skew-symmetric part of feed trough matrix.
    """
    # load matrices
    path = os.path.join(f'resources/poro-n{n}.mat')
    path = pkg_resources.resource_stream(__name__, path)

    data = loadmat(path)
    A = data['A']

    # parameter settings
    rho = 1e-3
    alpha = 0.79
    Minv = 7.80e3
    kappaNu = 633.33
    Rshift = 1e-3

    # update discretization matrices to reflect the parameter settings
    Y = rho * data['Y']
    D = alpha * data['D']
    M = Minv * data['M']
    K = kappaNu * data['K']
    Bp = data['Bp'].T
    Bf = data['Bf'].T

    # construct first-order port-Hamiltonian descriptor system
    # representation, similarly as in
    # R.Altmann, V.Mehrmann, B.Unger: Port - Hamiltonian
    # formulations of poroelastic network models, arXiv preprint
    # arXiv: 2012.01949, 2020.
    n = A.shape[0]
    m = M.shape[0]
    H = np.block([
        [Y, np.zeros((n, n + m))],
        [np.zeros((n, n)), A, np.zeros((n, m))],
        [np.zeros((m, n + n)), M]
    ])

    J = np.block([
        [np.zeros((n, n)), -A, D.T],
        [A, np.zeros((n, n + m))],
        [-D, np.zeros((m, n + m))]
    ])

    R = np.block([
        [np.zeros((n, 2 * n + m))],
        [np.zeros((n, 2 * n + m))],
        [np.zeros((m, 2 * n)), K]
    ]) + Rshift * np.eye(2 * n + m)

    G = np.block([
        [Bf, np.zeros((n, 1))],
        [np.zeros((n, 2))],
        [np.zeros((m, 1)), Bp]
    ])

    P = np.zeros(G.shape)
    S = np.zeros((G.shape[1], G.shape[1]))
    N = np.zeros((G.shape[1], G.shape[1]))
    Q = np.eye(J.shape[0])

    return H, J, R, G, P, S, N