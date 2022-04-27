import numpy as np


def lnorm(X, p=2, tau=1.0):
    r"""
    Approximates the :math:`\mathcal{L}^p` norm for discrete values.

    Parameters
    ----------
    X : numpy.ndarray
        Function values.
    p : int, optional
        Order. Default 2.
    tau : float, optional
        Stepsize. Default 1.0.

    Returns
    -------
    norm : float
        Norm.
    """
    if p == np.inf:
        return X.max()

    w = np.ones(X.shape[1]) * tau / 2
    w[0] = tau
    w[-1] = tau
    x = np.zeros(X.shape[1])
    for x_i in X:
        x += x_i ** p

    return np.inner(w, x) ** (1 / p)


if __name__ == '__main__':
    x = np.vstack((np.arange(1, 11), np.arange(10, 0, -1)))

    l2 = lnorm(x, p=2)
    print(f'|f(x)|_l2={l2:.2e}')
    linf = lnorm(x, p=np.inf)
    print(f'|f(x)|_linf={linf:.2e}')
