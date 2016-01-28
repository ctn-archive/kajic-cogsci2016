import os.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cPickle as pickle

import nengo
import data_processing.svd_utils
import numpy as np

from nengo import spa
from data_processing.read_association_matrix import load_vocabulary

def load_data(random=True):
    datapath = '../../data/processed/'
    pointers = np.load(datapath + 'was_256.npz')['W']
    #keys = np.loadtxt(datapath + 'words', dtype=np.str, delimiter=',').tolist()    
    with open(datapath + 'free_associations_vocabulary', 'rb') as f:
        id2voc = pickle.load(f)
        voc2id = pickle.load(f)
    #keys = keys[:-1]
    keys = [id2voc[i].upper() for i in range(len(pointers))]
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

    model.inp1 = spa.Input(pop1='CAKE')
    model.inp2 = spa.Input(pop2='SWISS')
    model.inp3 = spa.Input(pop3='COTTAGE')

    p1 = nengo.Probe(model.pop1.output, synapse=0.01)
    p2 = nengo.Probe(model.pop2.output, synapse=0.01)
    p3 = nengo.Probe(model.pop3.output, synapse=0.01)
    pt = nengo.Probe(model.target.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.5)
s1 = spa.similarity(sim.data[p1], spa_vocab)[300]
s2 = spa.similarity(sim.data[p2], spa_vocab)[300]
s3 = spa.similarity(sim.data[p3], spa_vocab)[300]
s = s1+s2+s3
st = spa.similarity(sim.data[pt], spa_vocab)[300]

a = np.argsort(s)[::-1]
b = np.argsort(st)[::-1]

print "Input to target:"
for x in a[:10]:
    print s[x], spa_vocab.keys[x]
print "Ouput from target:"
for x in b[:10]:
    print st[x], spa_vocab.keys[x]
print "Cheese should appear high in the list."
