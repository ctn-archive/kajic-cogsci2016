#### Semantic pointers

This directory contains semantic pointer representations of words generated using
different methods.

Naming convention for files in this directory:
`svd_5019w_128d_{mat,map}.{npz,pkl}`: `svd` describes the method used to get
the vectors, 5019w is the number of words (rows) in the matrix, 128d is the
dimensionality of the vector. `mat` is stored in `.npz` file and is a matrix
(numpy array), `map` is used to store mappings between words and indices
(word2ind) and mappings between indices and words (ind2word) in a `.pkl`.
