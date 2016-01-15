from nengo import spa


def filter_valid(rat_items, valid_words):
    for item in rat_items:
        cues_valid = all(w.lower() in valid_words for w in item.cues)
        target_valid = item.target.lower() in valid_words
        if cues_valid and target_valid:
            yield item


class Stimulus(object):
    def __init__(self, rat_items, item_duration=2.):
        self.rat_items = list(rat_items)
        self.item_duration = item_duration

    @property
    def n_items(self):
        return len(self.rat_items)

    @property
    def total_duration(self):
        return self.n_items * self.item_duration

    def t2idx(self, t):
        """Convert time to item index."""
        return t // self.item_duration

    def create_cue_fn(self, index):
        def cue(t):
            return self.rat_items[self.t2idx(t)].cues[index]
        return cue

    def target(self, t):
        return self.rat_items[self.t2idx(t)].target


class StimulusModule(spa.module.Module):
    def __init__(
            self, stimulus, d, vocab=None, label=None, seed=None,
            add_to_container=None):
        super(StimulusModule, self).__init__(label, seed, add_to_container)

        if vocab is None:
            vocab = d

        with self:
            self.cue1 = spa.State(d)
            self.cue2 = spa.State(d)
            self.cue3 = spa.State(d)

            self.stimulus = spa.Input(
                cue1=stimulus.create_cue_fn(0),
                cue2=stimulus.create_cue_fn(1),
                cue3=stimulus.create_cue_fn(2))

        self.outputs = dict(
            cue1=(self.cue1.output, vocab),
            cue2=(self.cue2.output, vocab),
            cue3=(self.cue3.output, vocab))
