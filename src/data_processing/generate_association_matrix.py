from __future__ import division
from __future__ import print_function

import cPickle as pickle
import gzip
import itertools
import os
import os.path
import string

from joblib import Parallel, delayed
import numpy as np


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


def gen_bigrams(words):
    id2word = list(words)
    word2id = {w: i for i, w in enumerate(id2word)}
    cs = itertools.product(string.ascii_lowercase, string.ascii_lowercase)

    path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data',
        'associationmatrices', 'google-bigrams')
    strength_mat = np.memmap(
        path, shape=(len(words), len(words)), mode='w+',
        dtype='uint32')
    strength_mat.fill(0.)
    Parallel(n_jobs=5)(
        delayed(process_bigram_file)(c1, c2, word2id, strength_mat)
        for c1, c2 in cs)
    return strength_mat, id2word, word2id


def process_bigram_file(c1, c2, word2id, output):
    template = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'raw',
        'google', 'googlebooks-eng-all-2gram-20120701-{c}.gz')
    filename = template.format(c=c1 + c2)
    print(filename)

    with gzip.open(filename, 'rt') as f:
        for line in f:
            ngram, year, count, _ = line.split('\t')
            if year != '2008':
                continue
            words = ngram.upper().split(' ')
            if any(w not in word2id for w in words):
                continue
            print(words)
            output[word2id[words[0]], word2id[words[1]]] = int(count)


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
        Filename without extension.
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


def load_assoc_mat(path, name):
    """Load an association matrix.

    Parameters
    ----------
    path : str
        Input directory
    name : str
        Filename without extension.

    Returns:
    --------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    strength_mat = np.load(mat_file)
    with open(map_file, 'rb') as f:
        id2word = pickle.load(f)
        word2id = pickle.load(f)

    return strength_mat, id2word, word2id
