class BaseModel:
    """
    The base class for all Ignis models

    Parameters
    ----------
    options: dict, optional
        Model-specific options
    """

    def __init__(self, options=None):
        pass

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
