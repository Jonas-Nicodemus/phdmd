import numpy as np

from pymor.models.iosys import LTIModel

from linalg.symmetric import sym, skew


class PHLTI(object):
    """
    Class representing a linear time invariant port-Hamiltonian system.
    """
    def __init__(self, E, J, R, G=None, P=None, D=None, Q=None):
        """
        Constructor.

        Parameters
        ----------
        E : numpy.ndarray
            Hamiltonian matrix.
        J : numpy.ndarray
            Skew-symmetric matrix.
        R : numpy.ndarray
            Dissipative matrix.
        G : numpy.ndarray, optional
            Input matrix. If `None` it is assumed that `J` and `R` contain already all other system matrices.
        P : numpy.ndarray
            Input matrix. If `None` it is assumed that `J` and `R` contain already all other system matrices.
        D : numpy.ndarray
            Feed trough matrix. If `None` it is assumed that `J` and `R` contain already all other system matrices.
        Q : numpy.ndarray
            Hamiltonian matrix. If `None` it is assumed that `Q` ist the identity i.e., `Q` already on the lhs,
            otherwise `Q` will be brought to the lhs by multiplying with `Q.T` from left.
        """
        if G is None:
            n = E.shape[0]
            P = R[:n, n:]
            S = R[n:, n:]

            G = J[:n, n:]
            N = J[n:, n:]

            R = R[:n, :n]
            J = J[:n, :n]
            D = S + N

        if Q is None:
            # Q already on lhs
            self.E = E
            self.J = J
            self.R = R
            self.G = G
            self.P = P
            self.D = D
        else:
            # bring Q on lhs
            self.E = Q.T @ E
            self.J = Q.T @ J @ Q
            self.R = Q.T @ R @ Q
            self.G = Q.T @ G
            self.P = Q.T @ P
            self.D = D

        self.S = sym(self.D)
        self.N = -skew(self.D)

        self.n, self.m = G.shape
        self.p = self.m

    def sim(self, U, T=None, x0=None):
        """
        Simulate the system for a given input signal and returns the resulting discrete input, state and output.

        Parameters
        ----------
        U : numpy.ndarray, callable
            Input signal as `numpy.ndarray` or `callable` object.
        T : numpy.ndarray, optional
            Time steps at which the input is defined and the output und state is returned.
            If `None` the input signal `U` is assumed to be an `numpy.ndarray` and the time steps are calculated from it
            with a step-size of 1.
        x0 : numpy.ndarray, optional
            Initial condition of the state. If `None` the initial condition is set to zero for all states.

        Returns
        -------
        U : numpy.ndarray
            Input values at the time steps `T`.
        X : numpy.ndarray
            States at the time steps `T`.
        Y : numpy.ndarray
            Output values at the time steps `T`.
        """
        if T is None:
            assert isinstance(U, np.ndarray)
            T = np.linspace(0, len(U))
        if not isinstance(U, np.ndarray):
            U = U(T)
            if U.ndim < 2:
                U = U[np.newaxis, :]

        if x0 is None:
            x0 = np.zeros(self.n)

        return self.implicit_midpoint(U, T, x0)

    def implicit_midpoint(self, U, T, x0):
        """
        Simulate the system via the implicit midpoint rule.

        Parameters
        ----------
        U : numpy.ndarray
            Input signal.
        T : numpy.ndarray
            Time steps at which the input is defined and the output und state is returned.
        x0 : numpy.ndarray
            Initial condition of the state.

        Returns
        -------
        U : numpy.ndarray
            Input values at the time steps `T`.
        X : numpy.ndarray
            States at the time steps `T`.
        Y : numpy.ndarray
            Output values at the time steps `T`.
        """
        delta = T[1] - T[0]

        M = (self.E - delta / 2 * (self.J - self.R))
        A = (self.E + delta / 2 * (self.J - self.R))

        X = np.zeros((self.n, len(T)))
        X[:, 0] = x0

        for i in range(len(T) - 1):
            U_midpoint = 1 / 2 * (U[:, i] + U[:, i + 1])
            X[:, i + 1] = np.linalg.solve(M, A @ X[:, i] + delta * (self.G - self.P) @ U_midpoint)

        Y = (self.G + self.P).T @ X + self.D @ U

        return U, X, Y

    def to_lti(self, matrices=True):
        """
        Converts the port-Hamiltonian linear time-invariant system in a standard linear time-invariant system.

        Parameters
        ----------
        matrices : bool, optional
            If `True` the lti matrices are returned, else a pymor `LTIModel`. Default `True`.

        Returns
        -------
        A : numpy.ndarray
            `A` matrix of the lti system.
        B : numpy.ndarray
            `B` matrix of the lti system.
        C : numpy.ndarray
            `C` matrix of the lti system.
        D : numpy.ndarray
            `D` matrix of the lti system.
        E : numpy.ndarray
            `E` matrix of the lti system.
        lti : `LTIModel`
            Instance of pyMOR `LTIModel`.
        """
        A = self.J - self.R
        B = self.G - self.P
        C = (self.G + self.P).T
        D = self.D
        E = self.E

        if matrices:
            return A, B, C, D, E
        else:
            return LTIModel.from_matrices(A, B, C, D, E=E)


if __name__ == "__main__":
    from model.msd import msd
    E, J, R, G, P, D, Q = msd()
    n = J.shape[0]
    x0 = np.zeros(n)
    u = lambda t: np.array([2 * np.sin(t ** 2), -np.cos(t ** 2)])
    T = np.linspace(0, 6, 500)
    ph = PHLTI(E, J, R, G, P, D, Q)

    U, X, Y = ph.sim(u, T, x0)
