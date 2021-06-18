"""
Topic modelling using a `tomotopy` LDA model.
"""

import os
import pathlib
import tempfile
import time
import uuid
import warnings

from tqdm.auto import tqdm

import ignis.util
from .base import BaseModel

tp = ignis.util.LazyLoader("tomotopy")

default_options = {
    "term_weighting": "pmi",
    "k": 10,
    "seed": 11399,
    # If "auto", will attempt to detect the number of available CPU cores and use
    # half that as the number of workers.
    "workers": "auto",
    "parallel_scheme": "default",
    "iterations": 2000,
    "update_every": 500,
    "until_max_ll": False,
    "until_max_coherence": False,
    "max_extra_iterations": 2000,
    "verbose": True,
    # --------------------------------------
    # Model options
    # Document-topic
    # N.B.: If hyper-parameter optimisation is on, setting this parameter manually
    # will not have much effect.
    "alpha": 0.1,
    # Topic-word
    # N.B.: This parameter can significantly affect the modelling results even if
    # optimisation is on.
    # Tomotopy's default is 0.01
    # If "auto", will use 1 / k
    "eta": "auto",
    # --------------------------------------
    # Automatic hyper-parameter optimisation
    # Number of burn-in iterations (Tomotopy's default is 0)
    "burn_in": 100,
    # Optimisation interval (set to 0 to turn off)
    "optim_interval": 10,
}


class LDAModel(BaseModel):
    """
    An Ignis model that wraps a `tomotopy` LDA model.

    The configurable keys for the `options` dictionary are described in the "Other
    Parameters" section.

    Parameters
    ----------
    corpus_slice: ignis.corpus.CorpusSlice
        The `ignis.corpus.CorpusSlice` to train the model on.
    options: dict, optional
        Model-specific options.

    Other Parameters
    ----------------
    term_weighting: {"idf", "pmi", "one"}
        Tomotopy term weighting scheme.
    k: int
        Number of topics to infer.
    seed: int
        Model random seed.
    workers: int or "auto"
        Number of worker processes to use.  If "auto", will use half the number of
        available CPU cores.
    parallel_scheme: {"partition", "copy_merge", "default", "none"}
        Tomotopy parallelism scheme.
    iterations: int
        Number of base sampling iterations to run.
    update_every: int
        How many iterations to run in a batch.

        If `verbose` is True, will print diagnostic information after each batch.
    until_max_ll: bool
        Whether or not to continue training until an approximate maximum per-word
        Log-Likelihood is reached.
    max_extra_iterations: int
        Limit on number of extra iterations to run if `until_max_ll` is True.
    verbose: bool
        Whether or not to print diagnostic information after each training batch.
    alpha: float
        Document-topic hyper-parameter for the Dirichlet distribution.
    eta: float or "auto"
        Topic-word hyper-parameter for the Dirichlet distribution.
        If "auto", will use `1 / k`.
    burn_in: int
        Burn-in iterations for hyper-parameter optimisation (The upstream default for
        `tomotopy` is 0)
    optim_interval: int
        Interval for automatic hyper-parameter optimisation (set to 0 to turn off)
    """

    def __init__(self, corpus_slice, options=None):
        super().__init__(corpus_slice, options)

        self.corpus_slice = corpus_slice
        self.model_type = "tp_lda"

        # `options` is a dict holding any user-defined model options.
        # Since it comes from an external source, we keep it as a separate dict rather
        # than pulling all the values in as instance variables.
        if options is None:
            options = {}

        # Start by filling in any missing options with the defaults.
        self.options = dict(default_options, **options)

        # Normalise options
        # -----------------
        # Term weighting
        term_weighting = self.options["term_weighting"]
        if term_weighting == "idf":
            tw = tp.TermWeight.IDF
        elif term_weighting == "pmi":
            tw = tp.TermWeight.PMI
        elif term_weighting == "one":
            tw = tp.TermWeight.ONE
        else:
            raise ValueError(
                "Invalid value for `term_weighting` (must be one of: idf, pmi, one)"
            )
        self.options["tw"] = tw

        # Worker count
        if self.options["workers"] == "auto":
            worker_count = int(os.cpu_count() / 2)
            # We need at least 1 worker
            self.options["workers"] = max(1, worker_count)

        # Parallel scheme
        parallel_scheme = self.options["parallel_scheme"]
        if parallel_scheme == "default":
            parallel = tp.ParallelScheme.DEFAULT
        elif parallel_scheme == "copy_merge":
            parallel = tp.ParallelScheme.COPY_MERGE
        elif parallel_scheme == "partition":
            parallel = tp.ParallelScheme.PARTITION
        elif parallel_scheme == "none":
            parallel = tp.ParallelScheme.NONE
        else:
            raise ValueError(
                "Invalid value for `parallel_scheme` (must be one of: default, "
                "copy_merge, partition, none)"
            )
        self.options["parallel"] = parallel

        # Eta
        if self.options["eta"] == "auto":
            eta = 1 / self.options["k"]
        else:
            eta = self.options["eta"]

        # Initialise model
        self.model = tp.LDAModel(
            tw=self.options["tw"],
            k=self.options["k"],
            seed=self.options["seed"],
            alpha=self.options["alpha"],
            eta=eta,
        )

        # When docs are added to `self.model`, we can only retrieve them from
        # `self.model.docs`, which acts like a list.  For bookkeeping, we keep track
        # of which Document (by ID) goes to which index within that list.
        self.doc_id_to_model_index = {}
        index = 0

        empty_docs = 0
        for doc_id in self.corpus_slice.document_ids:
            doc = self.corpus_slice.get_document(doc_id)

            # Because the Corpus stop list is dynamically applied, documents may have
            # empty token lists at this point
            if len(doc.tokens) == 0:
                empty_docs += 1
                continue

            self.model.add_doc(doc.tokens)
            self.doc_id_to_model_index[doc_id] = index
            index += 1

        if empty_docs > 0:
            warnings.warn(
                f"{empty_docs} document(s) skipped because they contain no tokens. "
                f"(The document(s) may have contained only stop words as tokens.)"
            )

        # Hyper-parameter optimisation
        self.model.burn_in = self.options["burn_in"]
        self.model.optim_interval = self.options["optim_interval"]

    @staticmethod
    def load_from_bytes(model_bytes):
        """
        Loads a `tomotopy.LDAModel` from its binary representation.

        Parameters
        ----------
        model_bytes: bytes
            The binary representation of the `tomotopy.LDAModel`.

        Returns
        -------
        tomotopy.LDAModel
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "load_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            with open(tmp_model_file, "wb") as fp:
                fp.write(model_bytes)

            # noinspection PyTypeChecker,PyCallByClass
            tp_model = tp.LDAModel.load(tmp_model_file)

        return tp_model

    def train(self):
        parallel = self.options["parallel"]
        workers = self.options["workers"]
        iterations = self.options["iterations"]
        update_every = self.options["update_every"]
        until_max_ll = self.options["until_max_ll"]
        until_max_coherence = self.options["until_max_coherence"]
        verbose = self.options["verbose"]

        origin_time = time.perf_counter()

        progress_bar = None
        if verbose:
            # print(
            #     f"Training model on {len(self.corpus_slice)} documents:\n"
            #     f"{self.options}\n",
            #     flush=True,
            # )
            progress_bar = tqdm(total=iterations, miniters=1)

        try:
            for i in range(0, iterations, update_every):
                self.model.train(update_every, workers=workers, parallel=parallel)
                if verbose:
                    current_coherence = self.get_coherence()
                    progress_bar.set_postfix({"Coherence": f"{current_coherence:.5f}"})
                    progress_bar.update(update_every)
                    # Let the progress bar update
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping train sequence.")

        if verbose:
            progress_bar.close()

        if until_max_ll:
            self._train_until_max_ll()

        if until_max_coherence:
            self._train_until_max_coherence()

        elapsed = time.perf_counter() - origin_time
        if verbose:
            print(
                f"Model training complete. ({elapsed:.3f}s)\n"
                f"\n"
                f"<Ignis Options>\n"
                f"| Workers: {workers}\n"
                f"| ParallelScheme: {repr(parallel)}\n"
                f"|"
            )
            self.model.summary(topic_word_top_n=10, flush=True)

    def _train_until_max_ll(self):
        """
        Extended training until an approximate maximum per-word LL is reached.

        Saves a temporary copy of the model before every iteration batch, and loads
        the last best model once LL stops increasing.
        """
        parallel = self.options["parallel"]
        workers = self.options["workers"]
        update_every = self.options["update_every"]
        max_extra_iterations = self.options["max_extra_iterations"]
        verbose = self.options["verbose"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "max_ll_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            self.model.save(tmp_model_file)

            if verbose:
                print(
                    "\n"
                    "Continuing to train until maximum log-likelihood.\n"
                    "(N.B.: This may not correlate with increased interpretability)\n",
                    flush=True,
                )
                progress_bar = tqdm(miniters=1)

            best_ll = self.model.ll_per_word
            i = 0
            batches_since_best = 0

            while True:
                try:
                    self.model.train(update_every, workers=workers, parallel=parallel)
                    i += update_every

                    current_ll = self.model.ll_per_word
                    if verbose:
                        progress_bar.set_postfix(
                            {"Log-likelihood": f"{current_ll:.5f}"}
                        )
                        progress_bar.update(update_every)
                        # Let the progress bar update
                        time.sleep(0.01)

                    if current_ll < best_ll:
                        batches_since_best += 1
                    else:
                        batches_since_best = 0
                        self.model.save(tmp_model_file)
                        best_ll = current_ll

                    if batches_since_best == 5 or i >= max_extra_iterations:
                        break

                except KeyboardInterrupt:
                    print("Stopping extended train sequence.")
                    break

            if verbose:
                progress_bar.close()

            # noinspection PyTypeChecker,PyCallByClass
            self.model = tp.LDAModel.load(tmp_model_file)

    def _train_until_max_coherence(self):
        """
        Extended training until an approximate best coherence score is reached.

        Saves a temporary copy of the model before every iteration batch, and loads
        the last best model once coherence stops improving.

        N.B.: Currently assumes that a higher coherence score is always better;
        should be true for u_mass, but might need to be checked if a different
        measure is used.
        """
        parallel = self.options["parallel"]
        workers = self.options["workers"]
        update_every = self.options["update_every"]
        max_extra_iterations = self.options["max_extra_iterations"]
        verbose = self.options["verbose"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "max_coherence_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            self.model.save(tmp_model_file)

            if verbose:
                print(
                    "\nContinuing to train until maximum coherence.\n",
                    flush=True,
                )
                progress_bar = tqdm(miniters=1)

            best_coherence = self.get_coherence()
            start_coherence = best_coherence
            i = 0
            batches_since_best = 0

            while True:
                try:
                    self.model.train(update_every, workers=workers, parallel=parallel)
                    i += update_every

                    current_coherence = self.get_coherence()
                    if verbose:
                        progress_bar.set_postfix(
                            {"Coherence": f"{current_coherence:.5f}"}
                        )
                        progress_bar.update(update_every)
                        # Let the progress bar update
                        time.sleep(0.01)

                    if current_coherence < best_coherence:
                        batches_since_best += 1
                    else:
                        batches_since_best = 0
                        self.model.save(tmp_model_file)
                        best_coherence = current_coherence

                    if batches_since_best == 5 or i >= max_extra_iterations:
                        break

                except KeyboardInterrupt:
                    print("Stopping extended train sequence.")
                    break

            if verbose:
                progress_bar.close()
                print(
                    f"Best coherence: {best_coherence:.5f} "
                    f"(Starting: {start_coherence:.5f})"
                )

            # noinspection PyTypeChecker,PyCallByClass
            self.model = tp.LDAModel.load(tmp_model_file)

    def get_num_topics(self):
        return self.model.k

    def get_topic_words(self, topic_id, top_n):
        # Tomotopy topics are 0-indexed
        tp_topic_id = topic_id - 1
        return self.model.get_topic_words(tp_topic_id, top_n)

    def get_topic_documents(self, topic_id, within_top_n):
        # N.B.: If calling consecutively for multiple topic IDs, this may be less
        # efficient than iterating over the documents directly and calling
        # `.get_document_topics()` instead.

        # Tomotopy topics are 0-indexed
        tp_topic_id = topic_id - 1

        topic_documents = []
        for doc_id in self.corpus_slice.document_ids:
            model_index = self.doc_id_to_model_index[doc_id]
            model_doc = self.model.docs[model_index]
            doc_topics = model_doc.get_topics(top_n=within_top_n)

            # Each item in doc_topics is a tuple of (<tp_topic_id>, <probability>)
            for topic, prob in doc_topics:
                if topic == tp_topic_id:
                    topic_documents.append((doc_id, prob))

        return topic_documents

    def get_document_topics(self, doc_id, top_n):
        if isinstance(doc_id, str):
            doc_id = uuid.UUID(doc_id)

        model_index = self.doc_id_to_model_index[doc_id]
        model_doc = self.model.docs[model_index]
        doc_topics = model_doc.get_topics(top_n=top_n)

        # Each item in doc_topics is a tuple of (<topic_id>, <probability>),
        # but the `topic_id` returned by Tomotopy is 0-indexed, so we need to add 1
        doc_topics = [(tp_topic_id + 1, prob) for tp_topic_id, prob in doc_topics]

        return doc_topics

    def get_document_top_topic(self, doc_id):
        return self.get_document_topics(doc_id, 1)[0]

    def get_coherence(self, coherence="c_npmi", top_n=30, **kwargs):
        """
        Use the `tomotopy.coherence` pipeline to get a coherence score for this
        trained model.

        Parameters
        ----------
        coherence: {"c_npmi", "c_v", "u_mass", "c_uci"}, optional
            Coherence measure to calculate.
            The four shorthand strings above can be used, or a custom combination of
            `tomotopy.coherence` classes can be used.
            N.B.: Unlike Gensim, the default is "u_mass", which is faster to calculate
        top_n: int, optional
            Number of top tokens to extract from each topic. The default of 30 matches
            the number of tokens shown per topic by `pyLDAvis`.

        Returns
        -------
        float
        """
        coherence_model = tp.coherence.Coherence(
            self.model, coherence=coherence, top_n=top_n, **kwargs
        )
        return coherence_model.get_score()

    def set_word_prior(self, word, prior_list):
        """
        This function sets the prior weight for the given word across each topic in
        the model.
        (contrib.: C. Ow)

        Parameters
        ----------
        word: str
            The word to set the word prior for.
        prior_list: iterable of float
            List of priors with length `k` (the number of topics).
        """
        k = self.get_num_topics()
        if len(prior_list) != k:
            raise ValueError(
                "Length mismatch between given word prior distribution and number of "
                "model topics."
            )
        else:
            self.model.set_word_prior(word, prior_list)

    def get_word_prior(self, word):
        """
        Gets the per-topic priors for the given word in the model.
        (contrib.: C. Ow)

        Parameters
        ----------
        word: str
            The word to get the word prior for.

        Returns
        -------
        iterable of float
            List of per-topic priors for the given word.
        """
        return self.model.get_word_prior(word)
