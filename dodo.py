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
        'targets': [
            os.path.join(
                'data', 'associationmatrices', 'freeassoc_symmetric' + ext)
            for ext in ['.npy', '.pkl']],
        'file_dep': [
            os.path.join('data', 'raw', 'freeassoc', file_['name'])
            for file_ in datasets['freeassoc']['files']],
    }
