import numpy as np
from itertools import chain
from data_processing.svd_utils import cos_sim


def load_problems():
    datapath = '../../data/processed/rat_items'
    items = np.loadtxt(datapath, dtype=np.character)

    return items


def load_vectors():
    datapath = '../../data/processed/rat_items'

    items = np.loadtxt(datapath, dtype=np.character).tolist()
    vocabulary = list(chain(*items))

    vectors = np.random.randn(len(vocabulary), 10)

    return vocabulary, vectors


def all_in_vocab(problem, vocabulary):
    """
    Check whether all problem cues and the target exist in the vocabulary. 
    Return True/False.
    """
    word_in_vocabulary = True

    for word in problem:
        word_in_vocabulary = word in vocabulary

    return word_in_vocabulary


# load RAT items as a list of lists
rat_problems = load_problems()

# load word vocabulary as list and vectors as an ndarray
vocabulary, vectors = load_vectors()

similarities_target = []
similarities_vocab = []

nr_problems = 0

for problem in rat_problems:
    if not all_in_vocab(problem, vocabulary):
        continue

    nr_problems += 1

    c1, c2, c3, target = problem
    cues = vectors[[vocabulary.index(c1),
                    vocabulary.index(c2),
                    vocabulary.index(c3)]]
    
    c1v = vectors[vocabulary.index(c1)]
    c2v = vectors[vocabulary.index(c2)]
    c3v = vectors[vocabulary.index(c3)]

    tv = vectors[vocabulary.index(target)]

    s1 = cos_sim(c1v, tv)
    s2 = cos_sim(c2v, tv)
    s3 = cos_sim(c3v, tv)

    sims = cos_sim(cues, tv)
    
    assert s1 == sims[0]
    assert s2 == sims[1]
    assert s3 == sims[2]

    similarities_target.append([s1, s2, s3])

sim_target = np.array(similarities_target, dtype=np.float)

print('Average similarity with the target:', sim_target.mean())
print('Average similarity with all words:', sim_target.mean())

