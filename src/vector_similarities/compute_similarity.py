import numpy as np
import cPickle as pickle
import os


def load_rat_problems(name='problems'):
    """Loads a list of 144 RAT items consisting of cues and targets.
    
    Parameters
    ----------
    name : str
        Text file containing RAT items

    Returns:
    --------
        A numpy array of characters.
    """
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        'data', 'rat')
    datapath = os.path.join(path, name + '.txt')

    items = np.loadtxt(datapath, dtype=np.character)

    return items


def load_vectors(name):
    """
    Function to load semantic pointers and mappings from the
    /data/semanticpointers/ directory.
    """
    matrix = 'associationmatrices'
    if 'svd' in name:
        matrix = 'semanticpointers'

    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        'data', matrix)

    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    vectors = np.load(mat_file)

    with open(map_file, 'r') as f:
        keys = pickle.load(f)
        w2i = pickle.load(f)

    return vectors, w2i, keys


def all_words_in_vocab(problem, vocabulary):
    """
    Check whether all problem cues and the target exist in the vocabulary.
    Return True/False.
    """
    word_in_vocabulary = True

    for word in problem:
        word_in_vocabulary = (word in vocabulary) and word_in_vocabulary

    return word_in_vocabulary


def get_similarities(W, w2i, prob, method):
    ''' Compute the similarity between the three cues and all words in the
    vocabulary.

    Parameters
    ----------
    W : ndarray
        Association matrix or matrix of vectors
    w2i : dict
        Word to id dictionary
    prob : list
        Cues and the target
    method : str
        Name of the method used to generate representations

    Returns:
    --------
    similarities : ndarray
        1D vector containing similarities to all words, averaged accross cues
    '''
    c1, c2, c3, target = prob
    cue_indices = [w2i[c1], w2i[c2], w2i[c3]]

    if 'svd' in method:
        sims = np.dot(W[cue_indices], W.T)
        similarities = sims.sum(axis=0)
    else:
        similarities = W[cue_indices]

    return similarities


def compute_similarity(method, quiet=True):
    '''For a given method, computes the similarity between all the words and
    the cues and finds the position of the target in sorted similarity values.

    Parameters
    ----------
    method : str
        Method used to create vectors.
    quiet : bool
        Print statistics

    Returns:
    --------
    sim_target : ndarray
        Similarity between the cues and the target for every RAT problem
    sim_everything : ndarray
        Similarity between the cues and all other words in vocabulary for every
        RAT problem
    targets : ndarray
        Position of target for every problem.
    '''
    rat_problems = load_rat_problems()
    vectors, w2i, words = load_vectors(method)

    similarities_target = []
    similarities_everything = []
    target_positions = []

    total_problems = len(rat_problems)

    for problem in rat_problems:
        if not all_words_in_vocab(problem, words):
            continue
        target = problem[-1]

        sims = get_similarities(vectors, w2i, problem, method)

        similarities_everything.append(sims.mean())
        similarities_target.append(sims[w2i[target]])

        # target position
        target_pos = lambda s, t: np.where(s.argsort()[::-1] == w2i[t])[0][0]
        target_positions.append(target_pos(sims, target))

    sim_target = np.array(similarities_target, dtype=np.float)
    sim_everything = np.array(similarities_everything, dtype=np.float)
    targets = np.array(target_positions, dtype=np.int)

    if not quiet:
        print('%d/%d problems exist with the %s vocabulary.' % (len(targets),
              total_problems, method))
        print('Average similarity with the target: %.5f (std=%.3f)' %
              (sim_target.mean(), sim_target.std()))
        print('Average similarity with all words: %.5f (std=%.3f)' %
              (sim_everything.mean(), sim_everything.std()))

    return sim_target, sim_everything, targets


if __name__ == '__main__':
    method = 'freeassoc_symmetric_svd_factorize_5018w_256d'
    
    st, se, targets = compute_similarity(method)
