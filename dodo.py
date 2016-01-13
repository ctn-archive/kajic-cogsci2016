import os.path

import scripts.fetch_data


def task_fetch_free_association_data():
    return {
        'actions': ['scripts/fetch_data.py freeassoc'],
        'targets': [
            os.path.join(
                'data', 'raw', 'freeassoc',
                scripts.fetch_data.extract_filename_from_url(url))
            for url in scripts.fetch_data.datasets['freeassoc']['files']],
        'uptodate': [True],  # only download data if not existent
    }
