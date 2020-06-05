import uuid


class Corpus:
    """
    A container for holding all the Documents relevant to a particular dataset.

    The same Corpus will be used even as sub-slices of the data go through iterative
    modelling -- Smaller sets of Documents will just be selected by ID.

    Attributes
    ----------
    documents: dict
        A mapping of Document IDs to the corresponding Documents.
    """

    def __init__(self):
        self.documents = {}

    def add_doc(self, metadata, tokens):
        """
        Creates a new Document with the given parameters and starts tracking it.

        Parameters
        ----------
        metadata: dict
            A general-purpose dictionary containing any metadata the user wants to
            track.

        tokens: iterable of str
            The individual content tokens in the given document.
        """
        this_doc = Document(metadata, tokens)
        self.documents[this_doc.id] = this_doc

    def slice_full(self):
        """
        Get a CorpusSlice containing all the documents in this Corpus.

        Returns
        -------
        CorpusSlice
        """
        return CorpusSlice(self, list(self.documents))


class CorpusSlice:
    """
    Contains some subset of the Documents in a Corpus, and keeps a reference to the
    root Corpus for bookkeeping and iteration.

    Parameters
    ----------
    root: Corpus
        The root Corpus instance for this slice.
    slice_ids: iterable of str
        The IDs for the documents in this slice.
    """

    def __init__(self, root, slice_ids):
        self.root = root
        self.documents = {}
        for slice_id in slice_ids:
            self.documents[slice_id] = root.documents[slice_id]


class Document:
    """
    Documents hold the textual content of each file in the Corpus, as well as any
    relevant metadata.

    Parameters
    ----------
    metadata: dict
        A general-purpose dictionary containing any metadata the user wants to
        track.

    tokens: iterable of str
        The individual content tokens in the given document.
    """

    def __init__(self, metadata, tokens):
        self.id = str(uuid.uuid4())
        self.metadata = metadata
        self.tokens = tokens

    def __str__(self):
        return f"ID: {self.id}\n\nMetadata: {self.metadata}\n\n{' '.join(self.tokens)}"
