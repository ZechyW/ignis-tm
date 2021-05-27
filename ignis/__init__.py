"""
.. include:: ../README.md
"""

import ignis.aurum
import ignis.corpus
import ignis.probat

from ignis.__version__ import __version__

Corpus = ignis.corpus.Corpus
load_corpus = ignis.corpus.load_corpus
load_slice = ignis.corpus.load_slice

train_model = ignis.probat.train_model
suggest_num_topics = ignis.probat.suggest_num_topics

load_results = ignis.aurum.load_results
