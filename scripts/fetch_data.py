#!/usr/bin/env python

from __future__ import print_function

import os
import os.path
import urllib
from urlparse import urlparse
import sys


datasets = {
    'freeassoc': {
        'description': 'University of South Florida Free Association Norms',
        'moreinfo': 'http://w3.usf.edu/FreeAssociation/',
        'files': [
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.A-B',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.C',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.D-F',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.G-K',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.L-O',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.P-R',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.S',
            'http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs.T-Z',
        ],
    },
}


def extract_filename_from_url(url):
    return os.path.basename(urlparse(url).path)


def report_progress(blocks_transferred, block_size, total_size):
    sys.stdout.write("\r{0}/{1} KiB".format(
        blocks_transferred * block_size // 1024, total_size // 1024))
    sys.stdout.flush()


def fetch(dataset_name, basepath=os.curdir):
    path = os.path.join(basepath, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    for url in datasets[dataset_name]['files']:
        target_filename = os.path.basename(urlparse(url).path)
        sys.stdout.write("Fetching " + target_filename + os.linesep)
        urllib.urlretrieve(
            url, os.path.join(path, target_filename), report_progress)
        sys.stdout.write(os.linesep)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch raw data files.",
        epilog="available datasets: " + ", ".join(datasets.keys()))
    parser.add_argument(
        'tofetch', nargs='*',
        help="Names of datasets to fetch. If not given, all will be fetched.")
    args = parser.parse_args()

    if len(args.tofetch) <= 0:
        args.tofetch = datasets.keys()

    target_dir = os.path.join(os.path.basename(
        __file__), os.pardir, 'data', 'raw')
    for dataset_name in args.tofetch:
        fetch(dataset_name, target_dir)
