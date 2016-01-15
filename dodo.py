import os.path

from src.datasets import datasets


def task_fetch_free_association_data():
    return {
        'actions': ['scripts/fetch_data.py freeassoc'],
        'targets': [
            os.path.join('data', 'raw', 'freeassoc', file_['name'])
            for file_ in datasets['freeassoc']['files']],
        'uptodate': [True],  # only download data if not existent
    }


def task_gen_symmetric_association_matrix():
    return {
        'actions': [
            'scripts/generate_association_matrix.py freeassoc symmetric'],
        'file_dep': [
            os.path.join('data', 'raw', 'freeassoc', file_['name'])
            for file_ in datasets['freeassoc']['files']],
        'targets': [
            os.path.join(
                'data', 'associationmatrices', 'freeassoc_symmetric' + ext)
            for ext in ['.npy', '.pkl']],
    }


def task_gen_semantic_pointers():
    return {
        'actions': [
            'scripts/generate_semantic_pointers.py freeassoc_symmetric '
            'svd_factorize 256'],
        'file_dep': [
            os.path.join(
                'data', 'associationmatrices', 'freeassoc_symmetric' + ext)
            for ext in ['.npy', '.pkl']],
        'targets': [
            os.path.join(
                'data', 'semanticpointers',
                'freeassoc_symmetric_svd_factorize_256w_256d' + ext)
            for ext in ['.npy', '.pkl']],
    }
