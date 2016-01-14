from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle
import os


def gen_asymmetric(words, association_database, diag=1.):
    """Generates an asymmetric association matrix from cues (rows) to targets
    (columns).

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    id2word = list(words)
    word2id = {w: i for i, w in enumerate(id2word)}
    strength_mat = np.zeros((len(words), len(words)))
    for cue, target, strength in association_database:
        strength_mat[word2id[cue], word2id[target]] += strength
    np.fill_diagonal(strength_mat, diag)
    assert np.all(strength_mat <= 1.)
    return strength_mat, id2word, word2id


def gen_symmetric(words, association_database, diag=1.):
    """Generates a symmetric association matrix from cues (rows) to targets
    (columns). This is done by using the asymmetric association matrix W and
    calculating (W + W.T)/2.

    Parameters
    ----------
    words : sequence
        Set of all occurring words.
    association_database : sequence of tuples (cue, target, strength)
        Associations to build matrix from.
    diag : float, optional
        Value to use for the diagonal (cue == target).

    Returns
    -------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    asymetric, id2word, word2id = gen_asymmetric(
        words, association_database, diag=diag)
    return (asymetric + asymetric.T) / 2., id2word, word2id


def print_stats(mat):
    assoc_counts = np.sum(mat > 0., axis=1)
    print(
        "Average number of associates per word:", np.mean(assoc_counts),
        np.std(assoc_counts))


def save_assoc_mat(path, name, strength_mat, id2word, word2id):
    """Save an association matrix.

    Parameters
    ----------
    path : str
        Output directory.
    name : str
        Filename without suffix.
    strength_mat : ndarray
        Association matrix.
    id2word: sequence/dict
        Mapping from index to word.
    word2id: dict
        Mapping from word to index.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    np.save(mat_file, strength_mat)
    with open(map_file, 'wb') as f:
        pickle.dump(id2word, f, protocol=2)
        pickle.dump(word2id, f, protocol=2)
