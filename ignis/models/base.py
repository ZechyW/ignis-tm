import ignis.corpus


class BaseModel:
    """
    The base class for all Ignis models.

    Also handles general CorpusSlice manipulation.

    NOTE: All Ignis models should have topic IDs that start from 1 and not 0;
    i.e., they should be in range(1, num_topics + 1)

    Parameters
    ----------
    corpus_slice: ignis.corpus.CorpusSlice
        The CorpusSlice to train the model on
    options: dict, optional
        Model-specific options
    """

    def __init__(self, corpus_slice, options=None):
        if not isinstance(corpus_slice, ignis.corpus.CorpusSlice):
            raise ValueError(
                "Ignis models must be instantiated with Corpus or "
                "CorpusSlice instances."
            )

        self.corpus_slice = corpus_slice

    def get_num_topics(self):
        """
        Returns
        -------
        int
            The number of topics in the trained model
        """
        pass

    def get_topic_words(self, topic_id, top_n):
        """
        NOTE: `topic_id` should start from 1 and not 0;
        i.e., it should be in `range(1, len(topics) + 1)`

        Parameters
        ----------
        topic_id
        top_n

        Returns
        -------
        iterable
            The `top_n` words in the topic `topic_id`, as a list of (<word:str>,
            <probability:float>)
        """
        pass

    def get_topic_documents(self, topic_id, within_top_n):
        """
        Find Documents that have Topic `topic_id` within its `n` most probable topics.

        Because topics are distributions over terms and documents are distributions
        over topics, documents don't belong to individual topics per se -- As such,
        we can consider a document to "belong" to a topic if it that topic is one of
        the `n` most probable topics for the document.

        This is especially significant if a term weighting scheme is used, because
        the most frequent words (i.e., what we usually consider stopwords) tend to be
        gathered into one or two topics, and we don't want to rule out a document if
        its single most probable topic is the stopword topic.

        NOTE: `topic_id` should start from 1 and not 0;
        i.e., it should be in `range(1, len(topics) + 1)`

        Parameters
        ----------
        topic_id
        within_top_n

        Returns
        -------
        iterable of tuples
            A list of tuples (<Document ID>, <topic_id probability>)
        """
        pass

    def get_document_topics(self, doc_id, top_n):
        """
        Get the top `n` most probable topics for the Document with the given ID.

        NOTE: The topic IDs in the results start from 1 and not 0.

        Parameters
        ----------
        doc_id
        top_n

        Returns
        -------
        iterable of tuples
            A list of tuples (<topic ID>, <probability>)
        """
        pass

    # ---------------------------------------------------------------------------------
    # Methods that operate on the CorpusSlice
    def get_document_by_id(self, doc_id):
        """
        Return the document from the model's CorpusSlice with the given ID.

        Parameters
        ----------
        doc_id

        Returns
        -------
        ignis.corpus.Document
        """
        return self.corpus_slice.documents[doc_id]
