import tomotopy as tp


class TomotopyLabeller:
    """
    Provides an interface to the Tomotopy First-order relevance labeller

    Parameters
    ----------
    model: tp.LDAModel
        A trained Tomotopy model
    verbose: bool, optional
        Whether or not to print verbose progress messages
    workers: int, optional
        Number of worker processes to use for the labeller -- Note that this *might*
        affect other random processes (e.g., the topic modelling proper) because it
        affects the parallelisation operations

    Other Parameters
    ----------------
    The other keyword arguments are labeller-specific options (See Tomotopy docs for
    details)
    """

    def __init__(
        self,
        model,
        min_cf=10,
        min_df=5,
        max_len=5,
        max_cand=10000,
        smoothing=0.01,
        mu=0.25,
        workers=8,
        verbose=False,
    ):
        self.labeller_type = "tomotopy"

        self.model = model

        # Bookkeeping: Save a record of the options used
        self.options = {
            "min_cf": min_cf,
            "min_df": min_df,
            "max_len": max_len,
            "max_cand": max_cand,
            "smoothing": smoothing,
            "mu": mu,
            "workers": workers,
            "verbose": verbose,
        }

        if verbose:
            print("Extracting label candidates from model...", flush=True)
        extractor = tp.label.PMIExtractor(
            min_cf=min_cf, min_df=min_df, max_len=max_len, max_cand=max_cand
        )
        candidates = extractor.extract(self.model)

        if verbose:
            print("Preparing First-order relevance labeller...", flush=True)
        self.labeller = tp.label.FoRelevance(
            self.model,
            candidates,
            min_df=min_df,
            smoothing=smoothing,
            mu=mu,
            workers=workers,
        )

        if verbose:
            print("Done.")

    def get_topic_labels(self, topic_id, top_n=10):
        """
        Gets the `top_n` labels from the saved Tomotopy FoRelevance labeller for the
        given `topic_id`.

        NOTE: `topic_id` should start from 1 and not 0;
        i.e., it should be in `range(1, len(topics) + 1)`

        Parameters
        ----------
        topic_id: int
            Topic ID to label
        top_n: int
            Number of labels to generate

        Returns
        -------
        tuple of [str, float]
            Returns (<label>, <score>) for each of the `top_n` labels for the topic.
        """
        # Tomotopy topics are 0-indexed
        tp_topic_id = topic_id - 1

        return self.labeller.get_topic_labels(k=tp_topic_id, top_n=top_n)
