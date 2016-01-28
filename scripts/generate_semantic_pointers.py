#!/usr/bin/env python

import os
import os.path
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, 'sparat'))

import argparse

from data_processing import dimension_reduction
from data_processing.generate_association_matrix import load_assoc_mat
from data_processing.spgen import from_assoc_matrix, save_pointers

parser = argparse.ArgumentParser(description="Generate semantic pointers.")
parser.add_argument(
    'association_matrix', nargs=1,
    help="Association matrix to base semantic pointers on.")
parser.add_argument(
    'dimred', nargs=1, help="Dimension reduction method.")
parser.add_argument(
    'd', nargs=1, type=int, help="Dimensionality of semantic pointers.")
args = parser.parse_args()

input_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'associationmatrices')
strength_mat, id2word, word2id = load_assoc_mat(
    input_dir, args.association_matrix[0])

pointers = from_assoc_matrix(
    strength_mat, getattr(dimension_reduction, args.dimred[0]), args.d[0])

output_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, 'data', 'semanticpointers')
name = '{assoc}_{dimred}_{words}w_{d}d'.format(
    assoc=args.association_matrix[0], dimred=args.dimred[0],
    words=pointers.shape[0], d=args.d[0])
save_pointers(output_dir, name, pointers, id2word, word2id)
