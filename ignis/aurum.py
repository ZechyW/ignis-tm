class Aurum:
    """
    Aurum instances hold the results of performing topic modelling over Documents.
    They provide methods for easily exploring the results and iterating over the topic
    modelling process.
    """

    def __init__(self, documents, model):
        """
        :param documents: Mapping of ids -> Documents
        :param model: Trained Tomotopy model
        """
        self.documents = documents
        self.model = model
