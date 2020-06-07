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
            print("Extracting label candidates from model...")
        extractor = tp.label.PMIExtractor(
            min_cf=min_cf, min_df=min_df, max_len=max_len, max_cand=max_cand
        )
        candidates = extractor.extract(self.model)

        if verbose:
            print("Preparing First-order relevance labeller...")
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

    def get_topic_labels(self, k, top_n=10):
        """
        Ensures the labeller is initialised, then passes the request through to the
        Tomotopy FoRelevance object

        Parameters
        ----------
        k: int
            Topic number to label
        top_n: int
            Number of labels to generate

        Returns
        -------
        tuple of [str, float]
            Returns (<label>, <score>) for each of the `top_n` labels for the topic.
        """
        return self.labeller.get_topic_labels(k=k, top_n=top_n)
