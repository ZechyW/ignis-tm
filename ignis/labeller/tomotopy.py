import tomotopy as tp

default_options = {
    # Candidates
    "min_cf": 10,
    "min_df": 5,
    "max_len": 5,
    "max_cand": 10000,
    # First-order Relevance
    "smoothing": 0.01,
    "mu": 0.25,
    "workers": 8,
}


class TomotopyLabeller:
    """
    Provides an interface to the Tomotopy First-order relevance labeller

    Parameters
    ----------
    model: tp.LDAModel
        A trained Tomotopy model
    options: dict, optional
        Labeller-specific options (See Tomotopy docs for details)
    """

    def __init__(self, model, options=None):
        if options is None:
            options = {}
        self.options = dict(default_options, **options)
        self.model = model

        # Initialise the labeller
        min_cf = self.options["min_cf"]
        min_df = self.options["min_df"]
        max_len = self.options["max_len"]
        max_cand = self.options["max_cand"]
        smoothing = self.options["smoothing"]
        mu = self.options["mu"]
        workers = self.options["workers"]

        extractor = tp.label.PMIExtractor(
            min_cf=min_cf, min_df=min_df, max_len=max_len, max_cand=max_cand
        )
        candidates = extractor.extract(self.model)

        self.labeller = tp.label.FoRelevance(
            self.model,
            candidates,
            min_df=min_df,
            smoothing=smoothing,
            mu=mu,
            workers=workers,
        )

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
