import os.path
import platform
import sys

import_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src'))
if import_path not in sys.path:
    sys.path.insert(0, import_path)

import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub

from model.benchmark import ConnecitonsRatModel

neurons_per_dimension=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
pspace = Param(
    neurons_per_dimension=neurons_per_dimension,
    seed=923)

min_items = 1
max_splits = 100

sharcnet_nodes = ['narwhal', 'bul', 'kraken', 'saw']
if any(platform.node().startswith(x) for x in sharcnet_nodes):
    workdir = '/work/jgosmann/rat'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '60m',
        'memory' : '6G',
    }

def execute(**kwargs):
    return RatModel().run(rmse=True, **kwargs)
