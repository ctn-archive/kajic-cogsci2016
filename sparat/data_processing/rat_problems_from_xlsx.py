"""
Reads the data from the Bowden experiment and stores only
those RAT items which exist in the free association database
"""

from __future__ import division

import numpy as np
import pandas as pd


def get_experimental_data():
    """
    Reads the spread sheet with tasks, solutions and human performance data
    stored on the disk.
    Returns two separate lists:
    rat_items: list of words
    performance: %corr (t=2), %corr (t=15), avg time (t=15), std time(t=15)
                 %corr (t=30)

    """

    path_rat = '../../data/raw/144CompoundBowden.xlsx'
    xls = pd.ExcelFile(path_rat)
    sheet_name = xls.sheet_names[0]
    id_c_cue = 0
    id_c_tar = 1
    id_c_2sec = 2
    id_c_15sec = 6
    id_c_15secm = 7
    id_c_15secstd = 8
    id_c_30sec = 9

    data = xls.parse(sheet_name, 0, index_col=None, na_values=['NA'],
                     parse_cols=[id_c_cue, id_c_tar, id_c_2sec,
                                 id_c_15sec, id_c_15secm, id_c_15secstd,
                                 id_c_30sec],
                     skip_footer=2)
    rat_items = []

    # 1.col: t=2 %
    # 2.col: t=15 % , 3.col: t=15 mean, 4.col: t=15 std
    # 5.col: t=30 %
    performance = np.zeros((len(data), 5))

    for i in range(len(data)):
        # extract cues and the target from the row
        cues = data.irow(i).Items.split('/')
        target = data.irow(i).Solutions.split('/')[0]
        rat_item = cues + [target]
        rat_item = [item.upper() for item in rat_item]

        # extract percentage of people solving the item
        performance[i, 0] = data.irow(i)[2]
        performance[i, 1] = data.irow(i)[3]
        performance[i, 4] = data.irow(i)[6]

        time = data.irow(i)[4]
        if type(time) not in (np.float, np.int):
            time = 15
        performance[i, 2] = time

        std = data.irow(i)[5]
        if type(std) not in (np.float, np.int):
            std = 0
        performance[i, 3] = std

        rat_items.append(rat_item)

    return rat_items, performance


def number_idx(array, number, sent=-1):
    pos = -1
    if number in array:
        pos = np.where(array == number)[0][0]
    return pos


if __name__ == "__main__":

    rat_items, performance = get_experimental_data()
    datapath = '../../data/rat/problems.txt'

    with open(datapath, 'w') as f:
        for idx, task in enumerate(rat_items):
            line = ' '.join(task)
            f.writelines(line + '\n')
        f.close()
        print('Succesfully stored RAT problems in:', datapath)
