import numpy as np


def cos_sim(v1, v2):
    """
    Compute cosine similarity between two vectors or matrices.
    """
    v1, v2 = np.atleast_2d(v1), np.atleast_2d(v2)

    dot = np.dot(v1, v2.T).flatten()
    norm = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)

    res = dot/norm

    return res
