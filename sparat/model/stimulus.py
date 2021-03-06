import nengo
from nengo import spa


def filter_valid(rat_items, valid_words):
    for item in rat_items:
        cues_valid = all(w in valid_words for w in item.cues)
        target_valid = item.target in valid_words
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
        return min(self.n_items - 1, int(t // self.item_duration))

    def create_cue_fn(self, index):
        def cue(t):
            return self.rat_items[self.t2idx(t)].cues[index]
        return cue

    def target(self, t):
        return self.rat_items[self.t2idx(t)].target


class StimulusModule(spa.module.Module):
    def __init__(
            self, stimulus, vocab, neurons_per_dimension, label=None, seed=None,
            add_to_container=None):
        super(StimulusModule, self).__init__(label, seed, add_to_container)

        d = vocab.dimensions

        def vocab_parse(fn):
            return lambda x: vocab.parse(fn(x)).v

        with self:
            self.cue1 = spa.State(
                d, subdimensions=64,
                neurons_per_dimension=neurons_per_dimension, vocab=vocab)
            self.cue2 = spa.State(
                d, subdimensions=64,
                neurons_per_dimension=neurons_per_dimension, vocab=vocab)
            self.cue3 = spa.State(
                d, subdimensions=64,
                neurons_per_dimension=neurons_per_dimension, vocab=vocab)

            self.cue1_input = nengo.Node(
                vocab_parse(stimulus.create_cue_fn(0)))
            nengo.Connection(self.cue1_input, self.cue1.input)

            self.cue2_input = nengo.Node(
                vocab_parse(stimulus.create_cue_fn(1)))
            nengo.Connection(self.cue2_input, self.cue2.input)

            self.cue3_input = nengo.Node(
                vocab_parse(stimulus.create_cue_fn(2)))
            nengo.Connection(self.cue3_input, self.cue1.input)

        self.outputs = dict(
            cue1=(self.cue1.output, vocab),
            cue2=(self.cue2.output, vocab),
            cue3=(self.cue3.output, vocab))
