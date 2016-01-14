import os
import os.path
import urllib
import sys


datasets = {
    'freeassoc': {
        'description': 'University of South Florida Free Association Norms',
        'moreinfo': 'http://w3.usf.edu/FreeAssociation/',
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
