import numpy as np

from scipy.linalg import block_diag


def msd(n=6, m=1, m_i=4, k_i=4, c_i=1, as_ph=True):
    """
    Returns the mass-spring-damper benchmark system (cf. :cite:`GugPBV12`), as port-Hamiltonian system.

    Parameters
    ----------
    n : int, optional
        Dimension of the state. Default 6.
    m : int, optional
        Dimension of the input resp. output. Default 2. (only 1 or 2 are implemented)
    m_i : float, optional
        Weight of the masses. Default 4.
    k_i : float, optional
        Stiffness of the springs. Default 4.
    c_i : float, optional
        Amount of damping. Default 1.

    Returns
    -------
    E : numpy.ndarray
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
    assert (n % 2) == 0

    n = int(n / 2)

    if m == 2:
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0], [0, 0, 0, 1 / m_i, 0, 0]])
    elif m == 1:
        B = np.array([[0, 1, 0, 0, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])
    else:
        assert False

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    J_i = np.array([[0, 1], [-1, 0]])
    J = np.kron(np.eye(3), J_i)
    R_i = np.array([[0, 0], [0, c_i]])
    R = np.kron(np.eye(3), R_i)

    G = B
    for i in range(4, n + 1):
        G = np.vstack((G, np.zeros((2, m))))
        C = np.hstack((C, np.zeros((m, 2))))
        J = block_diag(J, J_i)
        R = block_diag(R, R_i)

        A = block_diag(A, np.zeros((2, 2)))
        A[2 * i - 2, 2 * i - 2] = 0
        A[2 * i - 1, 2 * i - 1] = -c_i / m_i
        A[2 * i - 3, 2 * i - 2] = k_i
        A[2 * i - 2, 2 * i - 1] = 1 / m_i
        A[2 * i - 2, 2 * i - 3] = 0
        A[2 * i - 1, 2 * i - 2] = -2 * k_i
        A[2 * i - 1, 2 * i - 4] = k_i

    Q = np.linalg.solve(J - R, A)

    if not as_ph:
        return A, G, C, Q

    P = np.zeros(G.shape)
    S = np.zeros((m, m))
    N = np.zeros((m, m))
    E = np.eye(2 * n)

    # bring Q to left-hand side
    H = Q.T @ E
    J = Q.T @ J @ Q
    R = Q.T @ R @ Q
    G = Q.T @ G
    P = Q.T @ P
    Q = np.eye(J.shape[0])

    return H, J, R, G, P, S, N
