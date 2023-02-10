import numpy as np

from tqdm import tqdm

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel, PHLTIModel


def discretize(lti, U, T, x0, method='implicit_midpoint', return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system.

    Parameters
    ----------
    lti : pymor.models.iosys.PHLTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    method : str
        The method to use for the discretization. Available methods are:
        'implicit_midpoint', 'explicit_euler', 'explicit_midpoint', 'scipy_solve_ivp'.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    assert isinstance(lti, PHLTIModel) or isinstance(lti, LTIModel)
    if isinstance(lti, PHLTIModel):
        lti = lti.to_lti()

    match method:
        case 'implicit_midpoint':
            return implicit_midpoint(lti, U, T, x0, return_dXdt)
        case 'explicit_euler':
            return explicit_euler(lti, U, T, x0, return_dXdt)
        case 'explicit_midpoint':
            return explicit_midpoint(lti, U, T, x0, return_dXdt)
        case _:
            return scipy_solve_ivp(lti, U, T, x0, method, return_dXdt)


def implicit_midpoint(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the implicit midpoint method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    M = to_matrix(lti.E - delta / 2 * lti.A)
    AA = to_matrix(lti.E + delta / 2 * lti.A)
    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in tqdm(range(len(T) - 1)):
        U_midpoint = 1 / 2 * (U[:, i] + U[:, i + 1])
        X[:, i + 1] = np.linalg.solve(M, AA @ X[:, i] + delta * B @ U_midpoint)

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt


def explicit_euler(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the explicit Euler method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in tqdm(range(len(T) - 1)):
        X[:, i + 1] = X[:, i] + delta * np.linalg.solve(E, A @ X[:, i] + B @ U[:, i])

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt


def explicit_midpoint(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the explicit midpoint method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in tqdm(range(len(T) - 1)):
        X_ = X[:, i] + delta * np.linalg.solve(E, A @ X[:, i] + B @ U[:, i])
        X[:, i + 1] = X[:, i] + delta * np.linalg.solve(E, A @ X_ + B @ (1 / 2 * (U[:, i] + U[:, i + 1])))

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt


def scipy_solve_ivp(lti, u, T, x0, method='RK45', return_dXdt=False):
    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    U = u(T)
    if U.ndim < 2:
        U = U[np.newaxis, :]

    from scipy.integrate import solve_ivp

    def f(t, x, u):
        return np.linalg.solve(E, A @ x + B @ u(t))

    sol = solve_ivp(f, (T[0], T[-1]), x0, t_eval=T, method=method, args=(u,))
    X = sol.y
    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt
