"""
Base class for all `ignis` models.
Should never be instantiated directly.
"""

import ignis.corpus


class BaseModel:
    """
    The base class for all `ignis` models.

    Implemented child classes have the freedom to define their own default values for
    each method.

    **NOTE**: All `ignis` models should have topic IDs that are 1-indexed,
    not 0-indexed (i.e., they should be in `range(1, len(topics) + 1)`).

    Parameters
    ----------
    corpus_slice: ignis.corpus.CorpusSlice
        The `ignis.corpus.CorpusSlice` to train the model on.
    options: dict, optional
        Model-specific options.
    """

    def __init__(self, corpus_slice, *args, options=None, **kwargs):
        # Save a reference to the CorpusSlice we are modelling over
        self.corpus_slice = corpus_slice

        # Unique model type identifier for saving/loading results
        self.model_type = "<set_me>"

        # Model-specific options
        if options is None:
            options = {}
        self.options = options

        # The actual model, as created by the external topic modelling library
        self.model = None

    def train(self):
        """
        Trains the topic model with its configured options on its configured
        `ignis.corpus.CorpusSlice`.
        """
        raise NotImplementedError()

    def get_num_topics(self):
        """
        Get the number of topics in a trained topic model.

        Returns
        -------
        int
        """
        raise NotImplementedError()

    def get_topic_words(self, topic_id, top_n):
        """
        Get the `n` most probable words for a given topic.

        **NOTE**: `topic_id` is 1-indexed, not 0-indexed (i.e., it is in `range(1,
        len(topics) + 1)`).

        Parameters
        ----------
        topic_id: int
            The 1-indexed ID of the topic to consider.
        top_n: int
            The number of most probable words to return for the topic.

        Returns
        -------
        iterable of tuple
            A list of tuples of (`word`, `probability`).
        """
        raise NotImplementedError()

    def get_topic_documents(self, topic_id, within_top_n):
        """
        Find `ignis.corpus.Document` objects that have the topic with the given
        `topic_id` within their `n` most probable topics.

        Because topics are distributions over terms and documents are distributions
        over topics, documents don't belong to individual topics per se -- As such,
        we can consider a document to "belong" to a topic if it that topic is one of
        the `n` most probable topics for the document.

        This is especially significant if a term weighting scheme is used, because
        the most frequent words (i.e., what we usually consider stopwords) tend to be
        gathered into one or two topics, and we don't want to rule out a document if
        its single most probable topic is the stopword topic.

        **NOTE**: `topic_id` is 1-indexed, not 0-indexed (i.e., it is in `range(1,
        len(topics) + 1)`).

        Parameters
        ----------
        topic_id: int
            The 1-indexed ID of the topic to consider.
        within_top_n: int
            The top `n` topics per `ignis.corpus.Document` to look for `topic_id` in.

        Returns
        -------
        iterable of tuple
            A list of tuples of (`Document ID`, `topic_id probability`)
        """
        raise NotImplementedError()

    def get_document_topics(self, doc_id, top_n):
        """
        Get the top `n` most probable topics for the `ignis.corpus.Document` with the
        given ID.

        Children must accept either a string or UUID for `doc_id`.

        **NOTE**: `topic_id` is 1-indexed, not 0-indexed (i.e., it is in `range(1,
        len(topics) + 1)`).

        Parameters
        ----------
        doc_id: str or uuid.UUID
            The ID for the `ignis.corpus.Document`.
        top_n: int
            The number of top topics to return.

        Returns
        -------
        iterable of tuple
            A list of tuples of (`topic ID`, `probability`)
        """
        raise NotImplementedError()

    def get_document_top_topic(self, doc_id):
        """
        Get the most probable topic for a given `ignis.corpus.Document`.

        Children must accept either a string or UUID for `doc_id`.

        Parameters
        ----------
        doc_id: str or uuid.UUID
            ID for some `ignis.corpus.Document`.

        Returns
        -------
        int
            The 1-indexed ID for the given `ignis.corpus.Document` object's top topic.
        """
        raise NotImplementedError()

    def get_coherence(self, **kwargs):
        """
        Gets the current coherence score for a trained model.

        Parameters
        ----------
        **kwargs
            Children are free to define their own coherence calculation pipelines.

        Returns
        -------
        float
            The current coherence score for the model.
        """
        raise NotImplementedError()
