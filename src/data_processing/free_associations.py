import os

import pandas as pd

from ..datasets import datasets


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
        for row in df_normed.iterrows():
            cue, target = row['CUE'].lower(), row['TARGET'].lower()
            words.add(cue)
            words.add(target)

            strength = row[column]
            association_database.append((cue, target, strength))

    assert len(words) == 5018, "Number words should be 5018."
    assert normed_responses == 63619, \
        "Number of normed responses should be 63619"

    return words, association_database
