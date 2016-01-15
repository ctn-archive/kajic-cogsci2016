import numpy as np
from numpy.linalg import svd


def svd_factorize(W, nr_dim):
    """
    Return a low-dimensional representation of a matrix symmetric matrix W.

    Parameters
    ----------
    W : (N, N) ndarray
        Symmetric input matrix.
    nrdim : int
        Desired dimensionality.

    Returns
    -------
    r_mat : (N, nr_dim) ndarray
        Matrix factor with np.dot(r_mat, r_mat.T) approximately equal to W.

    """
    u, s, _ = svd(W)
    ur = u[:, :nr_dim]
    sr = np.diag(np.sqrt(s[:nr_dim]))
    return np.dot(ur, sr)
