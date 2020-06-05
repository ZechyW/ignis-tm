import pathlib
import tempfile
import time

import tomotopy as tp

import ignis.corpus
import ignis.models.base

default_options = {
    "term_weighting": "one",
    "k": 10,
    "seed": 11399,
    "workers": 8,
    "parallel_scheme": "default",
    "iterations": 1000,
    "update_every": 100,
    "until_max_ll": False,
    "max_extra_iterations": 5000,
    "verbose": False,
}


class LDAModel(ignis.models.base.BaseModel):
    """
    An Ignis model that performs LDA using Tomotopy.

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
    k: int
        Number of topics to infer
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
        super().__init__(options)

        # For saving/loading
        self.model_type = "tp_lda"

        if not isinstance(corpus_slice, ignis.corpus.CorpusSlice):
            raise ValueError(
                "Ignis models must be instantiated with Corpus or "
                "CorpusSlice instances."
            )

        self.corpus_slice = corpus_slice

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
        self.model = tp.LDAModel(
            tw=self.options["tw"], k=self.options["k"], seed=self.options["seed"]
        )

        # When docs are added to `self.model`, we can only retrieve them from
        # `self.model.docs`, which acts like a list.  For bookkeeping, we keep track
        # of which Document (by ID) goes to which index within that list.
        self.doc_id_to_model_index = {}
        index = 0
        for doc_id, doc in self.corpus_slice.documents.items():
            tokens = doc.tokens
            self.model.add_doc(tokens)
            self.doc_id_to_model_index[doc_id] = index
            index += 1

    def train(self):
        """
        Runs the modelling algorithm with the saved options and CorpusSlice.
        """
        parallel = self.options["parallel"]
        workers = self.options["workers"]
        iterations = self.options["iterations"]
        update_every = self.options["update_every"]
        until_max_ll = self.options["until_max_ll"]
        verbose = self.options["verbose"]

        origin_time = time.perf_counter()

        if verbose:
            print(f"Training LDA model:\n" f"{self.options}")
            print()

        try:
            for i in range(0, iterations, update_every):
                start_time = time.perf_counter()
                self.model.train(update_every, workers=workers, parallel=parallel)
                elapsed = time.perf_counter() - start_time
                if verbose:
                    print(
                        f"Iteration: {i + update_every}\t"
                        f"Log-likelihood: {self.model.ll_per_word}\t"
                        f"Time: {elapsed:.3f}s",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("Stopping train sequence.")

        if until_max_ll:
            self._train_until_max_ll()

        elapsed = time.perf_counter() - origin_time
        if verbose:
            print(f"Model training complete. ({elapsed:.3f}s)")

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
                print()
                print("Continuing to train until maximum log-likelihood.")
                print("(N.B.: This may not correlate with increased interpretability)")
                print()

            last_ll = self.model.ll_per_word
            i = 0
            consecutive_losses = 0

            while True:
                try:
                    start_time = time.perf_counter()
                    self.model.train(update_every, workers=workers, parallel=parallel)
                    i += update_every
                    elapsed = time.perf_counter() - start_time

                    current_ll = self.model.ll_per_word
                    if verbose:
                        print(
                            f"Iteration: {i}\t"
                            f"Log-likelihood: {current_ll}\t"
                            f"Time: {elapsed:.3f}s",
                            flush=True,
                        )

                    if current_ll < last_ll:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                        self.model.save(tmp_model_file)
                    last_ll = current_ll

                    if consecutive_losses == 2 or i >= max_extra_iterations:
                        break

                except KeyboardInterrupt:
                    print("Stopping extended train sequence.")
                    break

            # noinspection PyTypeChecker,PyCallByClass
            self.model = tp.LDAModel.load(tmp_model_file)

    def get_num_topics(self):
        return self.model.k

    def get_topic_words(self, topic_id, top_n):
        return self.model.get_topic_words(topic_id, top_n)
