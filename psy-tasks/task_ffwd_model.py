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

from model.benchmark import RatModel

n_subjects = 100
pspace = Param(
    exp_seed=np.arange(n_subjects),
    model_seed=np.arange(n_subjects, 2 * n_subjects))

min_items = 1
max_splits = n_subjects

sharcnet_nodes = ['narwhal', 'bull', 'kraken', 'saw']
if any(platform.node().startswith(x) for x in sharcnet_nodes):
    workdir = '/work/' + os.getlogin() + '/rat'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '60m',
        'memory' : '1024M',
    }

def execute(**kwargs):
    return RatModel().run(**kwargs)
