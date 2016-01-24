from __future__ import division

import numpy as np

import nengo
from nengo import spa


class FfwdRat(spa.module.Module):
    def __init__(
            self, d, vocab=None, label=None, seed=None, add_to_container=None):
        super(FfwdRat, self).__init__(label, seed, add_to_container)

        if vocab is None:
            vocab = d

        with self:
            self.cue1 = nengo.Node(size_in=d)
            self.cue2 = nengo.Node(size_in=d)
            self.cue3 = nengo.Node(size_in=d)
            self.rat_state = spa.State(d)

            nengo.Connection(
                self.cue1, self.rat_state.input, synapse=None,
                transform=1/3.)
            nengo.Connection(
                self.cue2, self.rat_state.input, synapse=None,
                transform=1/3.)
            nengo.Connection(
                self.cue3, self.rat_state.input, synapse=None,
                transform=1/3.)

        self.inputs = dict(
            cue1=(self.cue1, vocab),
            cue2=(self.cue1, vocab),
            cue3=(self.cue1, vocab))
        self.outputs = dict(default=(self.rat_state.output, vocab))


class FfwdConnectionsRat(spa.module.Module):
    def __init__(
            self, assoc, d, vocab, label=None, seed=None,
            add_to_container=None):
        super(FfwdConnectionsRat, self).__init__(label, seed, add_to_container)

        assoc = np.copy(assoc)
        assoc.fill_diagonal(0.)
        tr = np.dot(vocab.vectors.T, np.dot(assoc.T, vocab.vectors)) / 3.

        with self:
            self.cue1 = nengo.Node(size_in=d)
            self.cue2 = nengo.Node(size_in=d)
            self.cue3 = nengo.Node(size_in=d)
            self.rat_state = spa.State(d, vocab=vocab)

            nengo.Connection(
                self.cue1, self.rat_state.input, synapse=None,
                transform=tr)
            nengo.Connection(
                self.cue2, self.rat_state.input, synapse=None,
                transform=tr)
            nengo.Connection(
                self.cue3, self.rat_state.input, synapse=None,
                transform=tr)

        self.inputs = dict(
            cue1=(self.cue1, vocab),
            cue2=(self.cue1, vocab),
            cue3=(self.cue1, vocab))
        self.outputs = dict(default=(self.rat_state.output, vocab))
