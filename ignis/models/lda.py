import pathlib
import tempfile
import time

import tomotopy as tp
import tqdm

from .base import BaseModel

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


class LDAModel(BaseModel):
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

        for doc_id in self.corpus_slice.document_ids():
            doc = self.corpus_slice.get_document(doc_id)
            self.model.add_doc(doc.tokens)
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

        progress_bar = None
        if verbose:
            print(
                f"Training LDA model on {len(self.corpus_slice)} documents:\n"
                f"{self.options}\n",
                flush=True,
            )
            progress_bar = tqdm.tqdm(total=iterations, miniters=1)

        try:
            for i in range(0, iterations, update_every):
                self.model.train(update_every, workers=workers, parallel=parallel)
                if verbose:
                    progress_bar.set_postfix(
                        {"Log-likelihood": f"{self.model.ll_per_word:.5f}"}
                    )
                    progress_bar.update(update_every)
        except KeyboardInterrupt:
            print("Stopping train sequence.")

        if verbose:
            progress_bar.close()

        if until_max_ll:
            self._train_until_max_ll()

        elapsed = time.perf_counter() - origin_time
        if verbose:
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
                progress_bar = tqdm.tqdm(miniters=1)

            last_ll = self.model.ll_per_word
            i = 0
            consecutive_losses = 0

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

            if verbose:
                progress_bar.close()

            # noinspection PyTypeChecker,PyCallByClass
            self.model = tp.LDAModel.load(tmp_model_file)

    def get_num_topics(self):
        return self.model.k

    def get_topic_words(self, topic_id, top_n):
        # Tomotopy topics are 0-indexed
        tp_topic_id = topic_id - 1
        return self.model.get_topic_words(tp_topic_id, top_n)

    def get_topic_documents(self, topic_id, within_top_n):
        # Tomotopy topics are 0-indexed
        tp_topic_id = topic_id - 1

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
        model_index = self.doc_id_to_model_index[doc_id]
        model_doc = self.model.docs[model_index]
        doc_topics = model_doc.get_topics(top_n=top_n)

        # Each item in doc_topics is a tuple of (<topic_id>, <probability>),
        # but the `topic_id` returned by Tomotopy is 0-indexed, so we need to add 1
        doc_topics = [(tp_topic_id + 1, prob) for tp_topic_id, prob in doc_topics]

        return doc_topics