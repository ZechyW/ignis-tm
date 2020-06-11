import bz2
import pathlib
import pickle
import uuid
import collections


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
        self.documents = collections.OrderedDict()

    def add_doc(self, tokens, metadata=None, human_readable=None):
        """
        Creates a new Document with the given parameters and starts tracking it.

        Parameters
        ----------
        tokens: iterable of str
            The individual content tokens in the given document.
        metadata: dict, optional
            A general-purpose dictionary containing any metadata the user wants to
            track.
        human_readable: str, optional
            A human-readable version of the Document text.
            If None, will use the Document tokens joined with single spaces.

        Returns
        -------
        str
            The ID for the added Document
        """
        if metadata is None:
            metadata = collections.OrderedDict()
        if human_readable is None:
            human_readable = " ".join(tokens)
        doc = Document(tokens, metadata, human_readable)
        self.documents[doc.id] = doc
        return doc.id

    def save(self, filename):
        """
        Saves the Corpus object to the given file.
        Essentially uses a bz2-compressed Pickle format.

        Parameters
        ----------
        filename: str or pathlib.Path
            File to save the Corpus to
        """
        filename = pathlib.Path(filename)
        with bz2.open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def slice_full(self):
        """
        Get a CorpusSlice containing all the documents in this Corpus.

        Returns
        -------
        CorpusSlice
        """
        return CorpusSlice(root=self, slice_ids=list(self.documents))


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

    Attributes
    ----------
    documents: collections.OrderedDict
        Mapping of IDs to Documents.  Ordered by Document ID, but this shouldn't
        substantively affect the topic-modelling results.
    """

    def __init__(self, root, slice_ids):
        self.root = root
        self.documents = collections.OrderedDict()
        slice_ids.sort()
        for slice_id in slice_ids:
            self.documents[slice_id] = root.documents[slice_id]

    def __len__(self):
        return len(self.documents)

    def document_ids(self):
        return list(self.documents.keys())

    def get_document(self, doc_id):
        """
        Return the Document from this CorpusSlice with the given ID.

        Parameters
        ----------
        doc_id

        Returns
        -------
        Document
        """
        return self.documents[doc_id]

    def save(self, filename):
        """
        Saves the CorpusSlice object to the given file.
        Essentially uses a bz2-compressed Pickle format.

        Parameters
        ----------
        filename: str or pathlib.Path
            File to save the Corpus to
        """
        filename = pathlib.Path(filename)
        with bz2.open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def slice_by_ids(self, doc_ids):
        """
        Create a new CorpusSlice with the given Document IDs.
        The IDs do not have to be part of this CorpusSlice, as long as they are a
        part of the root Corpus.

        Parameters
        ----------
        doc_ids: iterable of str
            List of Document IDs

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if type(doc_ids) is str:
            raise RuntimeWarning(
                "Received a single string instead of an iterable of Document ID "
                "strings -- You probably did not intend to do this."
            )

        return CorpusSlice(self.root, doc_ids)

    def slice_by_tokens(self, tokens, include_root=False):
        """
        Create a new CorpusSlice with Documents that contain at least one of the
        given tokens.
        If `include_root` is True, will also search the root Corpus for Documents
        instead of limiting the search to the current CorpusSlice.

        Parameters
        ----------
        tokens: iterable of str
            A list of the tokens to search Documents for
        include_root: bool, optional
            Whether or not to search the root Corpus as well

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if type(tokens) is str:
            raise RuntimeWarning(
                "Received a single string instead of an iterable of token "
                "strings -- You probably did not intend to do this."
            )

        if include_root:
            search_docs = self.root.documents
        else:
            search_docs = self.documents

        search_tokens = set(tokens)

        found_doc_ids = []
        for doc_id, doc in search_docs.items():
            doc_tokens = set(doc.tokens)
            if len(search_tokens & doc_tokens) > 0:
                found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def concat(self, *other_slices):
        """
        Returns a new CorpusSlice that has the Documents from this instance and all
        the other specified CorpusSlices.

        Will retain the root Corpus from this instance.

        Parameters
        ----------
        other_slices: iterable of CorpusSlice

        Returns
        -------
        CorpusSlice
        """
        new_slice_ids = set(self.documents.keys())

        for other_slice in other_slices:
            if other_slice.root != self.root:
                raise RuntimeError(
                    "CorpusSlices can only be concatenated if they have the same root "
                    "Corpus."
                )

            slice_ids = set(other_slice.documents)
            new_slice_ids = new_slice_ids | slice_ids

        new_slice_ids = list(new_slice_ids)

        return CorpusSlice(self.root, new_slice_ids)


class Document:
    """
    Documents hold the textual content of each file in the Corpus, as well as any
    relevant metadata.

    Parameters
    ----------
    tokens: iterable of str
        The individual content tokens in the given document.
    metadata: dict
        A general-purpose dictionary containing any metadata the user wants to
        track.
    human_readable: str
        A string representing the Document in human-readable form.
    """

    def __init__(self, tokens, metadata, human_readable):
        self.id = str(uuid.uuid4())
        self.tokens = tokens
        self.metadata = metadata
        self.human_readable = human_readable

    def __str__(self):
        return (
            f"ID: {self.id}\n\nMetadata: {self.metadata}\n\n" f"{self.human_readable}"
        )


def load_corpus(filename):
    """
    Loads a Corpus object from the given file.

    Conceptually, Corpus objects contain the full amount of data for a given dataset.

    Some subset of the Corpus (up to the full Corpus itself) must be sliced into a
    CorpusSlice to perform topic modelling over the data, and these CorpusSlices can
    be iteratively expanded or contracted freely within the full set of Corpus data.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the Corpus object from.

    Returns
    -------
    ignis.corpus.Corpus
    """
    with bz2.open(filename, "rb") as fp:
        loaded = pickle.load(fp)

    if not isinstance(loaded, Corpus):
        raise ValueError(f"File does not contain a Corpus object: '{filename}'")

    return loaded


def load_slice(filename):
    """
    Loads a CorpusSlice object from the given file.

    CorpusSlices contain a specific subset of some root Corpus, and can be passed
    directly as input into topic models.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the CorpusSlice object from.

    Returns
    -------
    ignis.corpus.CorpusSlice
    """
    with bz2.open(filename, "rb") as fp:
        loaded = pickle.load(fp)

    if not isinstance(loaded, CorpusSlice):
        raise ValueError(f"File does not contain a CorpusSlice object: '{filename}'")

    return loaded
