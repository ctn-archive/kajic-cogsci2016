import os
import os.path
import urllib
import sys

import pandas as pd


def load_free_association_data(path, column='FSG'):
    """Loads the free association data from `path`.

    Parameters
    ----------
    path : str
        Path to load data from.
    column : str, optional
        Column in the data files that gives the association strength. Use 'FSG'
        for forward strength and 'BSG' for backward strength.

    Returns
    -------
    A tuple (words, association_database) where words is a set of all occuring
    words (either as cue or target) and association_database is a list of
    associations. Each association is a tuple (cue, target, strength).
    """
    normed_responses = 0
    words = set()
    association_database = []

    filenames = [f['name'] for f in datasets['freeassoc']['files']]
    for filename in filenames:
        df = pd.read_csv(
            os.path.join(path, filename), skipinitialspace=True,
            comment='<')  # comment='<' is a hackish way to skip HTML tags
        df[column] = pd.to_numeric(df[column])

        df_normed = df[df['NORMED?'] == 'YES']
        normed_responses += len(df_normed)

        # extract norms
        for _, row in df_normed.iterrows():
            cue, target = row['CUE'].lower(), row['TARGET'].lower()
            words.add(cue)
            words.add(target)

            strength = row[column]
            association_database.append((cue, target, strength))

    assert len(words) == 5018, "Number words should be 5018."
    assert normed_responses == 63619, \
        "Number of normed responses should be 63619"

    return words, association_database


datasets = {
    'freeassoc': {
        'description': 'University of South Florida Free Association Norms',
        'moreinfo': 'http://w3.usf.edu/FreeAssociation/',
        'loader': load_free_association_data,
        'files': [
            {
                'name': 'Cue_Target_Pairs.A-B',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.A-B',
            }, {
                'name': 'Cue_Target_Pairs.C',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.C',
            }, {
                'name': 'Cue_Target_Pairs.D-F',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.D-F',
            }, {
                'name': 'Cue_Target_Pairs.G-K',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.G-K',
            }, {
                'name': 'Cue_Target_Pairs.L-O',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.L-O',
            }, {
                'name': 'Cue_Target_Pairs.P-R',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.P-R',
            }, {
                'name': 'Cue_Target_Pairs.S',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.S',
            }, {
                'name': 'Cue_Target_Pairs.T-Z',
                'url':
                    'http://w3.usf.edu/FreeAssociation/AppendixA/'
                    'Cue_Target_Pairs.T-Z',
            }
        ],
    },
}


def report_progress(blocks_transferred, block_size, total_size):
    sys.stdout.write("\r{0}/{1} KiB".format(
        blocks_transferred * block_size // 1024, total_size // 1024))
    sys.stdout.flush()


def fetch(dataset_name, basepath=os.curdir):
    path = os.path.join(basepath, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    for file_ in datasets[dataset_name]['files']:
        sys.stdout.write("Fetching " + file_['name'] + os.linesep)
        urllib.urlretrieve(
            file_['url'], os.path.join(path, file_['name']), report_progress)
        sys.stdout.write(os.linesep)
