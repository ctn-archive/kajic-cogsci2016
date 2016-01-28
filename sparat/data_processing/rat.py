from collections import namedtuple

RatItem = namedtuple('RatItem', ['id', 'cues', 'target'])


def load_rat_items(path):
    items = []

    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            words = line.split()
            items.append(RatItem(i, words[:3], words[3]))

    return items
