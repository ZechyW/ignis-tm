import collections
import pathlib
import tempfile
import time
import uuid
import warnings

from tqdm.auto import tqdm

import ignis.util
from .base import BaseModel

tp = ignis.util.LazyLoader("tomotopy")
gensim = ignis.util.LazyLoader("gensim")

default_options = {
    "term_weighting": "one",
    "initial_k": 2,
    "seed": 11399,
    "workers": 8,
    "parallel_scheme": "default",
    "iterations": 500,
    "update_every": 100,
    "until_max_ll": False,
    "until_max_coherence": False,
    "max_extra_iterations": 1000,
    "verbose": False,
    # HDP model options
    # Document-table
    "alpha": 0.1,
    # Topic-word
    "eta": 0.01,
    # Table-topic
    "gamma": 0.1,
}


class HDPModel(BaseModel):
    """
    An Ignis model that wraps a Tomotopy HDP model.

    Parameters
    ----------
    corpus_slice: ignis.corpus.CorpusSlice
        The CorpusSlice to train the model on
    options: dict, optional
        Model-specific options

    Notes
    -----
    `options` can contain any of the following key-value pairs:

    term_weighting: {"idf", "pmi", "one"}
        Tomotopy term weighting scheme
    initial_k: int
        Number of initial topics; the total number of topics will be adjusted
        automatically as the model is trained
    seed: int
        Model random seed
    workers: int
        Number of worker processes to use
    parallel_scheme: {"partition", "copy_merge", "default", "none"}
        Tomotopy parallelism scheme
    iterations: int
        Number of base sampling iterations to run
    update_every: int
        How many iterations to run in a batch -- If `verbose` is True, will print
        diagnostic information after each batch
    until_max_ll: bool
        Whether or not to continue training until an approximate maximum per-word
        LL is reached
    max_extra_iterations: int
        Limit on number of extra iterations to run if `until_max_ll` is True
    verbose: bool
        Whether or not to print diagnostic information after each training batch
    """

    def __init__(self, corpus_slice, options=None):
        super().__init__(corpus_slice, options)

        self.corpus_slice = corpus_slice
        self.model_type = "tp_hdp"

        # `options` is a dict holding any user-defined model options.
        # Since it comes from an external source, we keep it as a separate dict rather
        # than pulling all the values in as instance variables.
        if options is None:
            options = {}

        # Start by filling in any missing options with the defaults.
        self.options = dict(default_options, **options)

        # Normalise options
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

        # Initialise model
        self.model = tp.HDPModel(
            tw=self.options["tw"],
            initial_k=self.options["initial_k"],
            seed=self.options["seed"],
            alpha=self.options["alpha"],
            eta=self.options["eta"],
            gamma=self.options["gamma"],
        )

        # When docs are added to `self.model`, we can only retrieve them from
        # `self.model.docs`, which acts like a list.  For bookkeeping, we keep track
        # of which Document (by ID) goes to which index within that list.
        self.doc_id_to_model_index = {}
        index = 0

        for doc_id in self.corpus_slice.document_ids():
            doc = self.corpus_slice.get_document(doc_id)
            self.model.add_doc(doc.tokens)
            self.doc_id_to_model_index[doc_id] = index
            index += 1

        # After a HDP model is trained, some number of topics might not be alive; we
        # should ignore these topics when dealing with user input.
        # User topic starts from 1; model_topic is in range (0, k), including dead
        # topics
        self.user_topic_to_model_topic = {}
        self.model_topic_to_user_topic = {}

    @staticmethod
    def load_from_bytes(model_bytes):
        """
        Loads a Tomotopy LDAModel from its binary representation

        Parameters
        ----------
        model_bytes: bytes

        Returns
        -------
        tp.LDAModel
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "load_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            with open(tmp_model_file, "wb") as fp:
                fp.write(model_bytes)

            # noinspection PyTypeChecker,PyCallByClass
            tp_model = tp.HDPModel.load(tmp_model_file)

        return tp_model

    def train(self):
        """
        Runs the modelling algorithm with the saved options and CorpusSlice.
        """
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
            print(
                f"Training model on {len(self.corpus_slice)} documents:\n"
                f"{self.options}\n",
                flush=True,
            )
            progress_bar = tqdm(total=iterations, miniters=1)

        try:
            for i in range(0, iterations, update_every):
                self.model.train(update_every, workers=workers, parallel=parallel)
                if verbose:
                    current_coherence = self.get_coherence(processes=workers)
                    progress_bar.set_postfix(
                        {
                            "Coherence": f"{current_coherence:.5f}",
                            "Num topics": self.model.live_k,
                        }
                    )
                    progress_bar.update(update_every)
                    # To allow the tqdm bar to update, if in a Jupyter notebook
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping train sequence.")

        if verbose:
            progress_bar.close()

        if until_max_ll:
            self._train_until_max_ll()

        if until_max_coherence:
            self._train_until_max_coherence()

        # Check for live topics, map to user topics
        self.user_topic_to_model_topic = {}
        self.model_topic_to_user_topic = {}
        current_user_topic = 1
        for tp_topic_id in range(self.model.k):
            if self.model.is_live_topic(tp_topic_id):
                self.user_topic_to_model_topic[current_user_topic] = tp_topic_id
                current_user_topic += 1
        for user_topic, model_topic in self.user_topic_to_model_topic.items():
            self.model_topic_to_user_topic[model_topic] = user_topic

        elapsed = time.perf_counter() - origin_time
        if verbose:
            print(
                f"Docs: {len(self.model.docs)}, "
                f"Vocab size: {len(self.model.used_vocabs)}, "
                f"Total Words: {self.model.num_words}"
            )
            print(f"Model training complete. ({elapsed:.3f}s)", flush=True)

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
                        # To allow the tqdm bar to update, if in a Jupyter notebook
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
            self.model = tp.HDPModel.load(tmp_model_file)

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
                    "\n" "Continuing to train until maximum coherence.\n", flush=True,
                )
                progress_bar = tqdm(miniters=1)

            best_coherence = self.get_coherence(processes=workers)
            start_coherence = best_coherence
            i = 0
            batches_since_best = 0

            while True:
                try:
                    self.model.train(update_every, workers=workers, parallel=parallel)
                    i += update_every

                    current_coherence = self.get_coherence(processes=workers)
                    if verbose:
                        progress_bar.set_postfix(
                            {
                                "Coherence": f"{current_coherence:.5f}",
                                "Num topics": self.model.live_k,
                            }
                        )
                        progress_bar.update(update_every)
                        # To allow the tqdm bar to update, if in a Jupyter notebook
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
            self.model = tp.HDPModel.load(tmp_model_file)

    def get_num_topics(self):
        return self.model.live_k

    def get_topic_words(self, topic_id, top_n):
        # Tomotopy topics are 0-indexed
        tp_topic_id = self.user_topic_to_model_topic[topic_id]
        return self.model.get_topic_words(tp_topic_id, top_n)

    def get_topic_documents(self, topic_id, within_top_n):
        """
        If calling consecutively for multiple topic IDs, this is less efficient than
        iterating over the documents directly and calling `.get_document_topics()`
        instead.

        Parameters
        ----------
        topic_id
        within_top_n

        Returns
        -------

        """
        # Tomotopy topics are 0-indexed
        tp_topic_id = self.user_topic_to_model_topic[topic_id]

        topic_documents = []
        for doc_id in self.corpus_slice.document_ids():
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
        doc_topics = [
            (self.model_topic_to_user_topic[tp_topic_id], prob)
            for tp_topic_id, prob in doc_topics
        ]

        return doc_topics

    def get_coherence(self, coherence="u_mass", top_n=30, processes=8):
        """
        Use Gensim's `models.coherencemodel` to get a coherence score for a trained
        LDAModel.

        Parameters
        ----------
        coherence: {"u_mass", "c_v", "c_uci", "c_npmi"}, optional
            Coherence measure to calculate.
            N.B.: Unlike Gensim, the default is "u_mass", which is faster to calculate
        top_n: int, optional
            Number of top words to extract from each topic. The default of 30 matches
            the number of words shown per topic by pyLDAvis
        processes: int, optional
            Number of processes to use for probability estimation phase

        Returns
        -------
        float
        """
        with warnings.catch_warnings():
            # At time of coding, Gensim 3.8.0 is the latest version available on the
            # main Anaconda repo, and it triggers DeprecationWarnings when the
            # CoherenceModel is used in this way
            warnings.simplefilter("ignore", category=DeprecationWarning)

            topics = []
            for k in range(self.model.k):
                if not self.model.is_live_topic(k):
                    continue

                word_probs = self.model.get_topic_words(k, top_n)
                topics.append([word for word, prob in word_probs])

            texts = []
            corpus = []
            for doc in self.model.docs:
                words = [self.model.vocabs[token_id] for token_id in doc.words]
                texts.append(words)
                freqs = list(collections.Counter(doc.words).items())
                corpus.append(freqs)

            id2word = dict(enumerate(self.model.vocabs))
            dictionary = gensim.corpora.dictionary.Dictionary.from_corpus(
                corpus, id2word
            )

            cm = gensim.models.coherencemodel.CoherenceModel(
                topics=topics,
                texts=texts,
                corpus=corpus,
                dictionary=dictionary,
                coherence=coherence,
                topn=top_n,
                processes=processes,
            )

            # For debugging the interface between Tomotopy and Gensim
            # coherence = cm.get_coherence()
            # return {
            #     "coherence": coherence,
            #     "topics": topics,
            #     "texts": texts,
            #     "corpus": corpus,
            #     "dictionary": dictionary,
            # }
            return cm.get_coherence()
