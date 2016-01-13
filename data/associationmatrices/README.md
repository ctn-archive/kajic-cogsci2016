#### Association Matrices

Pickled files, each containing three elements:
- association word strength matrix 
- mappings of indices to words (`dict`)
- mappings of words to indices (`dict`)

Because the association matrices can be derived in different ways (e.g. human
data, web-scraping) it is useful name them according to their source and use
the README to provide more information.

Current data:
`association_norms_symm`: Association matrix derived from the [Free
Association Norms, University of Florida](http://w3.usf.edu/FreeAssociation/).
The original matrix is not symmetric, this one has been created by adding the
transpose of the original. Generated with:
`/data_processing/generate_association_matrix.py`


