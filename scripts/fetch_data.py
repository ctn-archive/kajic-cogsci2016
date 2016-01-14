#!/usr/bin/env python

import os
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

import argparse

from datasets import datasets, fetch


parser = argparse.ArgumentParser(
    description="Fetch raw data files.",
    epilog="available datasets: " + ", ".join(datasets.keys()))
parser.add_argument(
    'tofetch', nargs='*',
    help="Names of datasets to fetch. If not given, all will be fetched.")
args = parser.parse_args()

if len(args.tofetch) <= 0:
    args.tofetch = datasets.keys()

target_dir = os.path.join(os.path.dirname(
    __file__), os.pardir, 'data', 'raw')
for dataset_name in args.tofetch:
    fetch(dataset_name, target_dir)
