import numpy as np
import cPickle as pickle

from data_processing.svd import cos_sim


def load_rat_problems():
    datapath = '../../data/rat/problems.txt'
    items = np.loadtxt(datapath, dtype=np.character)

    return items


def load_vectors(filename):
    """
    Function to load semantic pointers and mappings from the
    /data/semanticpointers/ directory.
    """
    path = '../../data/semanticpointers/'
    datapath = path + filename

    mapping_path = datapath + '_map.pkl'
    vectors_path = datapath + '_mat.npz'

    npvec = np.load(vectors_path)
    vectors = npvec[npvec.keys()[0]]

    with open(mapping_path, 'r') as f:
        w2i = pickle.load(f)
        i2w = pickle.load(f)

    return vectors, w2i, i2w


def all_words_in_vocab(problem, vocabulary):
    """
    Check whether all problem cues and the target exist in the vocabulary.
    Return True/False.
    """
    word_in_vocabulary = True

    for word in problem:
        word_in_vocabulary = (word in vocabulary) and word_in_vocabulary

    return word_in_vocabulary



method = 'random_483w_16d'
method = 'svd_5018w_16d'

# load RAT items as a list of lists
rat_problems = load_rat_problems()

# load word vocabulary as list and vectors as an ndarray
vectors, w2i, i2w = load_vectors(method)

similarities_target, similarities_everything = [], []

nr_problems = 0
total_problems = len(rat_problems)

for problem in rat_problems:
    # check the RAT words exist in the vocabulary
    if not all_words_in_vocab(problem, w2i.keys()):
        continue

    nr_problems += 1

    c1, c2, c3, target = problem

    # get vectors for cues and target
    cue_vectors = vectors[[w2i[c1], w2i[c2], w2i[c3]]]
    target_vector = vectors[w2i[target]]

    # cues -> target similarity
    cue_target_sim = cos_sim(cue_vectors, target_vector)
    similarities_target.append(cue_target_sim)

    # cues -> all other words similarity
    cues_words_sim = [cos_sim(cue_vectors[0], vectors).mean(),
                      cos_sim(cue_vectors[1], vectors).mean(),
                      cos_sim(cue_vectors[2], vectors).mean()]
    similarities_everything.append(cues_words_sim)

sim_target = np.array(similarities_target, dtype=np.float)
sim_everything = np.array(similarities_everything, dtype=np.float)

print('%d/%d problems exist with the %s vocabulary.' % (nr_problems,
      total_problems, method))
print('Average similarity with the target: %.5f (std=%.3f)' %
      (sim_target.mean(), sim_target.std()))
print('Average similarity with all words: %.5f (std=%.3f)' %
      (sim_everything.mean(), sim_everything.std()))
