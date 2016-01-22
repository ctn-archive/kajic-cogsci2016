import os.path

from src.datasets import datasets


def assocmat_files(name):
    assocdir = os.path.join('data', 'associationmatrices')
    return [os.path.join(assocdir, name + ext) for ext in ['.npy', '.pkl']]


def sp_files(name):
    spdir = os.path.join('data', 'semanticpointers')
    return [os.path.join(spdir, name + ext) for ext in ['.npy', '.pkl']]


def task_fetch_data():
    for name, dataset in datasets.items():
        yield {
            'name': name,
            'actions': ['scripts/fetch_data.py ' + name],
            'targets': [
                os.path.join('data', 'raw', name, file_['name'])
                for file_ in dataset['files']],
            'uptodate': [True],  # only download data if not existent
        }


def task_gen_association_matrices():
    matrices = [
        ('freeassoc', 'symmetric'),
        ('freeassoc', 'asymmetric'),
        ('google', '1grams'),
        ('google', 'bigrams')]
    for dataset, method in matrices:
        yield {
            'name': dataset + '_' + method,
            'actions': [
                'scripts/generate_association_matrix.py {0} {1}'.format(
                    dataset, method)],
            'file_dep': [
                os.path.join('data', 'raw', dataset, file_['name'])
                for file_ in datasets[dataset]['files']],
            'targets': assocmat_files(dataset + '_' + method),
        }

    yield {
        'name': 'google_combined',
        'actions': [
            'scripts/transform_association_matrix.py add google_combined '
            'google_bigrams google_1grams'],
        'file_dep': assocmat_files('google_bigrams') + assocmat_files(
            'google_1grams'),
        'targets': assocmat_files('google_combined'),
    }

    yield {
        'name': 'google_normalized',
        'actions': [
            'scripts/transform_association_matrix.py normalize '
            'google_normalized google_combined'],
        'file_dep': assocmat_files('google_combined'),
        'targets': assocmat_files('google_normalized'),
    }

    yield {
        'name': 'google_symmetric',
        'actions': [
            'scripts/transform_association_matrix.py symmetrify '
            'google_symmetric google_normalized'],
        'file_dep': assocmat_files('google_normalized'),
        'targets': assocmat_files('google_symmetric'),
    }


def task_svd():
    for assocmat in ['freeassoc_symmetric', 'google_symmetric']:
        for d in [4096, 3062, 2048, 1024, 768, 512, 380, 256, 128]:
            yield {
                'name': assocmat + '_' + str(d),
                'actions': [
                    'scripts/generate_semantic_pointers.py {0} svd_factorize '
                    '{1}'.format(assocmat, d)],
                'file_dep': assocmat_files(assocmat),
                'targets': sp_files(
                    assocmat + '_svd_factorize_5018w_' + str(d) + 'd'),
            }
