import numpy as np


def svd(A, hermitian=False, variant=None, tol=1e-12, t=None):
    """
    Wrapper around `numpy.svd` to perform the four variants of the singular value decomposition
    (https://en.wikipedia.org/wiki/Singular_value_decomposition).

    Parameters
    ----------
    A : numpy.ndarray
        A real or complex matrix to perform the svd on.
    hermitian : bool, optional
        If True, `A` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.
    variant : str, optional
        Variant of the svd.  Should be one of

            - 'full'
            - 'thin'
            - 'skinny'
            - 'trunc'
    tol : float, optional
        In the case of 'skinny' the zero tolerance and in the case of 'trunc' the truncation tolerance.
    t : int, optional
        In the case of 'trunc' the truncation index.

    Returns
    -------
    U : numpy.ndarray
        Left singular vectors.
    s : numpy.ndarray
        Singular values.
    V : numpy.ndarray
        Right singular vectors.
    """
    U, s, Vh = np.linalg.svd(A, hermitian=hermitian)
    V = Vh.conj().T

    match variant:
        case 'full':
            pass
        case 'thin':
            n, m = A.shape
            k = min(n, m)
            U = U[:, :k]
            V = V[:, :k]
        case 'skinny':
            r = np.argmax(s / s[0] < tol)
            r = r if r > 0 else len(s)
            U = U[:, :r]
            V = V[:, :r]
            s = s[:r]
        case 'trunc':
            t = np.argmax(s / s[0] < tol) if t is None else t
            t = t if t > 0 else len(s)
            U = U[:, :t]
            V = V[:, :t]
            s = s[:t]
        case None:
            pass

    return U, s, V


if __name__ == "__main__":
    n = 10
    m = 100
    A = np.random.randint(10, size=(n, m))
    A[-1] = 0
    r = np.linalg.matrix_rank(A)
    assert r == 9

    U, s, V = svd(A)
    assert U.shape == (n, n)
    assert V.shape == (m, m)
    assert len(s) == n

    U, s, V = svd(A, variant='thin')
    assert U.shape == (n, n)
    assert V.shape == (m, n)
    assert len(s) == n

    U, s, V = svd(A, variant='skinny')
    assert U.shape == (n, r)
    assert V.shape == (m, r)
    assert len(s) == r

    t = 5
    U, s, V = svd(A, variant='trunc', t=t)
    assert U.shape == (n, t)
    assert V.shape == (m, t)
    assert len(s) == t








