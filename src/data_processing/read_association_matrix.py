"""
Script containing useful helper functions to deal with processed free
association data.
"""

from __future__ import division

import numpy as np
import cPickle as pickle
import pdb

def load_vocabulary():
    """
    Unpickle and return the content of the file
    `free_associations_vocabulary` generated by `process_data.py`.

    Output
    ------
        W:          association matrix
        id2voc:     dictionary, keys are word ids and values words
        voc2id:     dictionary, keys are words and values word ids
    """

    # absolute path to the free association norms data
    path = '/home/ivana/phd/workspace/spa_rat/data/processed/'
    filename = 'free_associations_vocabulary'

    try:
        with open(path+filename, 'rb') as f:
            id2voc = pickle.load(f)
            voc2id = pickle.load(f)
            Wsparse = pickle.load(f)
    except IOError:
        raise IOError('Association matrix "' + filename  + '" not found' +\
                      ' in ' + path + '. To generate the matrix run ' +\
                      'generate_association_matrix.py')

    # convert to dense matrix (stored as sparse for memory reasons)
    W = np.asarray(Wsparse.todense())

    # normalize weights to [0-1] interval
    W /= W.max()

    return W, id2voc, voc2id


def show_associates(word):
    """
    Show the associates of a word in descending order of association strength.
    """
    W, ids, voc = load_vocabulary()

    associated_ids = np.asarray(W[voc[word]].nonzero()[0])
    strengths = W[voc[word], associated_ids]

    sorted_strengths_id = strengths.argsort()[::-1]

    print('Words associated to', word.upper())

    for idx in sorted_strengths_id:
        print ids[associated_ids[idx]], associated_ids[idx], strengths[idx]

    return associated_ids[sorted_strengths_id]

load_vocabulary()
