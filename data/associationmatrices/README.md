#### Association Matrices

Each association matrix is stored with two files:

1. `.npy` file in NumPy format containing the actual association matrix.
2. `.pkl` file in Python pickle format containing to Python objects: the mapping
   of indices to words and the mapping of words to indices (in this order).

The current naming scheme is `<dataset>_<method>` where `<dataset>` is the
dataset on which the association strengths are based on and `<method>` is how
the association matrix was obtained from the dataset.

Current data:
`freeassoc_symmetric`: Association matrix derived from the [Free
Association Norms, University of Florida](http://w3.usf.edu/FreeAssociation/).
The original matrix is not symmetric, this one has been created by adding the
transpose of the original. Generated with:
`/data_processing/generate_association_matrix.py`


