import numpy as np
import cPickle as pickle

wordspath = '../../data/rat/words.csv'
sppath = '../../data/semanticpointers/'

items = np.loadtxt(wordspath, dtype=np.character).tostring()
words = items.split(',')

nr_dim = 16
nr_w = len(words)
idx = range(nr_w)
name = 'random_%dw_%dd_' % (nr_w, nr_dim)


# store vectors in range [-1, 1]
vectors = 2*np.random.rand(nr_w, nr_dim)-1
np.savez(name + 'mat', vectors)

# store mappings
w2i = dict(zip(words, idx))
i2w = dict(zip(idx, words))

with open(sppath + name + 'map.pkl', 'w') as f:
    pickle.dump(w2i, f)
    pickle.dump(i2w, f)

print('Dumped %s in %s' % (name[:-1], sppath))
