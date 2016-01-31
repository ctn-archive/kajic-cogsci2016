#!/usr/bin/env python

import os
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import random
from random import shuffle

import ctn_benchmark
import nengo
from nengo import spa
import numpy as np

from data_processing.generate_association_matrix import load_assoc_mat
from data_processing.spgen import load_pointers
from data_processing.rat import load_rat_items
from model.stimulus import filter_valid, Stimulus, StimulusModule
from model.ffwd import FfwdRat, FfwdConnectionsRat


class RatModel(ctn_benchmark.Benchmark):
    def params(self):
        self.default("number of dimensions", d=256)
        self.default(
            "semantic pointer file",
            sp_file='freeassoc_symmetric_svd_factorize_5018w_256d')
        self.default("experiment seed", exp_seed=23)
        self.default("model seed", model_seed=42)

    def model(self, p):
        random.seed(p.exp_seed)

        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'data')
        sp_path = os.path.join(data_dir, 'semanticpointers')
        pointers, i2w, _ = load_pointers(sp_path, p.sp_file)

        rat_path = os.path.join(data_dir, 'rat', 'problems.txt')
        self.rat_items = list(filter_valid(load_rat_items(rat_path), i2w))
        shuffle(self.rat_items)

        with spa.SPA(seed=p.model_seed) as model:
            # set up vocab
            vocab = model.get_default_vocab(p.d)
            for i, pointer in enumerate(pointers):
                vocab.add(i2w[i].upper(), pointer)

            # set up model
            self.stimulus = Stimulus(self.rat_items)
            model.stimulus = StimulusModule(self.stimulus, p.d)
            model.rat_model = FfwdRat(p.d)
            nengo.Connection(model.stimulus.cue1.output, model.rat_model.cue1)
            nengo.Connection(model.stimulus.cue2.output, model.rat_model.cue2)
            nengo.Connection(model.stimulus.cue3.output, model.rat_model.cue3)
            self.p_output = nengo.Probe(model.rat_model.rat_state.output)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.stimulus.total_duration)
        return dict(
            output=sim.data[self.p_output],
            rat_items=[x.id for x in self.rat_items])


class ConnectionsRatModel(ctn_benchmark.Benchmark):
    def params(self):
        self.default("neurons per dimension", neurons_per_dimension=50)
        self.default("number of dimensions", d=5120)
        self.default(
            "association matrix", assocmat='freeassoc_asymmetric')
        self.default(
            "semantic pointer file",
            sp_file='freeassoc_asymmetric_randomize_orthonormal_5018w_5120d')
        self.default("rat problems", ratfile='example.txt')
        self.default("experiment seed", exp_seed=23)
        self.default("model seed", model_seed=42)
        self.default("running to record rmse", rmse=False)

    def model(self, p):
        random.seed(p.exp_seed)

        data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'data')
        sp_path = os.path.join(data_dir, 'associationmatrices')
        assoc, i2w, _ = load_assoc_mat(sp_path, p.assocmat)

        sp_path = os.path.join(data_dir, 'semanticpointers')
        pointers, _, _ = load_pointers(sp_path, p.sp_file)

        rat_path = os.path.join(data_dir, 'rat', p.ratfile)
        self.rat_items = list(filter_valid(load_rat_items(rat_path), i2w))

        with spa.SPA(seed=p.model_seed) as model:
            model.config[nengo.Connection].solver = nengo.solvers.LstsqL2(
                solver=nengo.solvers.randomized_svd)

            self.model = model
            # set up vocab
            self.vocab = model.get_default_vocab(p.d)
            for i, pointer in enumerate(pointers):
                sanitized = i2w[i].upper().replace(' ', '_').replace(
                    '+', '_').replace('-', '_').replace('&', '_').replace(
                        "'", '_')
                self.vocab.add(sanitized, pointer)

            # set up model
            self.stimulus = Stimulus(self.rat_items)
            model.stimulus = StimulusModule(
                self.stimulus, self.vocab, p.neurons_per_dimension)
            model.rat_model = FfwdConnectionsRat(
                assoc, self.vocab,
                neurons_per_dimension=p.neurons_per_dimension)
            nengo.Connection(model.stimulus.cue1.output, model.rat_model.cue1)
            nengo.Connection(model.stimulus.cue2.output, model.rat_model.cue2)
            nengo.Connection(model.stimulus.cue3.output, model.rat_model.cue3)
            self.p_output = nengo.Probe(
                model.rat_model.rat_state.output, synapse=0.003)
            self.p_cue1 = nengo.Probe(
                model.stimulus.cue1.output, synapse=0.003)
            self.p_cue2 = nengo.Probe(
                model.stimulus.cue2.output, synapse=0.003)
            self.p_cue3 = nengo.Probe(
                model.stimulus.cue3.output, synapse=0.003)
            self.p_spikes = nengo.Probe(
                model.rat_model.rat_state.state_ensembles.ensembles[0].neurons,
                'spikes')

            tr = np.dot(vocab.vectors.T, np.dot(assoc.T, vocab.vectors)) / 3.
            direct_result = nengo.Ensemble(
                n_neurons=1, dimensions=p.d, neuron_type=nengo.Direct())
            nengo.Connection(
                model.stimulus.cue1.output, direct_result, transform=tr)
            nengo.Connection(
                model.stimulus.cue2.output, direct_result, transform=tr)
            nengo.Connection(
                model.stimulus.cue3.output, direct_result, transform=tr)
            self.p_direct = nengo.Probe(direct_result, synapse=0.003)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.stimulus.total_duration)
        if p.rmse:
            m_sim = np.root(np.mean(np.square(
                sim.data[self.p_direct] - sim.data[self.p_output])))
            result = dict(m_sim=m_sim)
        else:
            sort_indices = nengo.utils.ensemble.sorted_neurons(
                self.model.rat_model.rat_state.state_ensembles.ensembles[0],
                sim, 1000)
            result = dict(
                trange=sim.trange(),
                output=sim.data[self.p_output],
                cue1=sim.data[self.p_cue1],
                cue2=sim.data[self.p_cue2],
                cue3=sim.data[self.p_cue3],
                rat_items=[x.id for x in self.rat_items],
                vocab_keys=self.vocab.keys,
                vocab_vectors=self.vocab.vectors,
                spikes=sim.data[self.p_spikes],
                sort_indices=sort_indices)
            data_dir = os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, 'data')
            np.savez(
                os.path.join(data_dir, 'FfwdConnectionsRat.npz'), **result)
        return result

if __name__ == '__main__':
    print nengo.__file__
    ConnectionsRatModel().run()
