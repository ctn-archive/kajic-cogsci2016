from scipy.linalg import svd
import cPickle as pickle
import numpy as np
import os
from data_processing.read_association_matrix import load_vocabulary


def cos_sim(v1, v2):
    """
    Compute cosine similarity between two vectors or matrices.
    """
    v1, v2 = np.atleast_2d(v1), np.atleast_2d(v2)

    dot = np.dot(v1, v2.T).flatten()
    norm = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)

    res = dot/norm

    return res


def reduced_matrix(W, nr_dim):
    """
    Return a low-dimensional representation of a matrix W.

    Input
    -----
        W:          N x N matrix
        nrdim:      desired dimensionality

    Output
    ------
        r_mat:      N x nr_dim matrx

    """
    u, s, v = svd(W)
    nr_word = len(W)

    ur = u[:, :nr_dim]
    sr = np.diag(np.sqrt(s[:nr_dim]))
    # vr = v[:nrdim, :]

    r_mat = np.dot(ur, sr)

    norms = np.linalg.norm(r_mat, axis=1)

    for i in range(nr_word):
        if norms[i] > 0:
            r_mat[i] /= float(norms[i])

    return r_mat


if __name__ == "__main__":
    W, i2w, w2i = load_vocabulary()
    nr_word = len(i2w)
    nr_dim = 16

    path = '../../data/semanticpointers/'
    was = reduced_matrix(W, nr_dim=nr_dim)

    if os.path.isdir(path):
        # save vectors
        filename = 'svd_%dw_%dd' % (nr_word, nr_dim)
        np.savez(path + filename + '_mat', W=was)

        # save mappings
        with open(path + filename + '_map.pkl', 'w') as f:
            up = lambda d: map(lambda x: x.upper(), d)
            w2i_up = dict(zip(up(w2i.keys()), w2i.values()))
            i2w_up = dict(zip(i2w.keys(), up(i2w.values())))
            pickle.dump(w2i_up, f)
            pickle.dump(i2w_up, f)

        print('Saved matrix and mapping for %s!' % (path+filename))
    else:
        print('Computed, but not saved because %s does not exist!' % path)
