import nengo
import data_processing.svd_utils
import numpy as np

from nengo import spa
from data.raw.freeassociations.read_data import load_vocabulary

def load_data(random=True):
    datapath = '../../data/processed/'
    pointers = np.load(datapath + 'was_128.npz')['W']
    keys = np.loadtxt(datapath + 'words', dtype=np.str, delimiter=',').tolist()    
    keys = keys[:-1]
    return keys, pointers



keys, pointers = load_data()
nr_words, nr_dim = pointers.shape
assert len(keys) == nr_words

spa_vocab = spa.Vocabulary(dimensions=nr_dim)

for key, p in zip(keys, pointers):
    spa_vocab.add(key, p)
    
with spa.SPA() as model:
    model.pop1 = spa.State(dimensions=nr_dim, vocab=spa_vocab)
    model.pop2 = spa.State(dimensions=nr_dim, vocab=spa_vocab)
    model.pop3 = spa.State(dimensions=nr_dim, vocab=spa_vocab)
    
    model.target = spa.State(dimensions=nr_dim, vocab=spa_vocab)
    
    nengo.Connection(model.pop1.output, model.target.input)
    nengo.Connection(model.pop2.output, model.target.input)
    nengo.Connection(model.pop3.output, model.target.input)

    model.inp1 = spa.Input(pop1='0')
    model.inp2 = spa.Input(pop2='0')
    model.inp3 = spa.Input(pop3='0')
    
    
    
    
    
    
    
    
