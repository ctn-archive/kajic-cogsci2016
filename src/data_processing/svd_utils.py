from scipy.linalg import svd
import numpy as np
import os

from read_association_matrix import load_vocabulary


def cos_sim(v1, v2):
    """
    Compute cosine similarity between two vectors.
    """
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    res = 0

    if norm > 0:
        res = np.dot(v1, v2)/norm

    return res


def reduced_matrix(W, nr_dim, path=''):
    """
    Return a low-dimensional representation of a matrix W.

    Input
    -----
        W:          N x N matrix
        nrdim:      desired dimensionality
        path:       directory path for saving the svd-reduced matrix W

    Output
    ------
        r_mat:      N x nr_dim matrx

    """
    u, s, v = svd(W)
    nr_word = len(W)

    ur = u[:, :nr_dim]
    sr = np.eye(nr_dim)*s[:nr_dim]
    # vr = v[:nrdim, :]

    r_mat = np.dot(ur, sr)

    norms = np.linalg.norm(r_mat, axis=1)

    for i in range(nr_word):
        if norms[i] > 0:
            r_mat[i] /= float(norms[i])
    
    if os.path.isdir(path):
        filename = 'was_' + str(nr_dim)
        np.savez(path+filename, W=r_mat)
        print('Saved:' + path+filename)

    return r_mat


if __name__=="__main__":
    W, _, _ = load_vocabulary()

    path = '/home/ivana/phd/workspace/spa_rat/data/processed/'
    was = reduced_matrix(W, nr_dim=256, path=path)
    print('done')

