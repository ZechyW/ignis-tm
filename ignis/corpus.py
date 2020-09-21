import bz2
import collections
import json
import pathlib
import pickle
import re
import uuid

from bs4 import BeautifulSoup


class Corpus:
    """
    A container for holding all the Documents relevant to a particular dataset.

    The same Corpus will be used even as sub-slices of the data go through iterative
    modelling -- Smaller sets of Documents will just be selected by ID.

    Corpora Documents are tracked by insertion order, but CorpusSlices are shuffled
    (viz., they are sorted by the randomly-generated Document IDs).

    Attributes
    ----------
    documents: dict
        A mapping of Document IDs to the corresponding Documents.
    """

    def __init__(self):
        self.documents = collections.OrderedDict()

    def add_doc(self, tokens, metadata=None, display_str=None, plain_text=None):
        """
        Creates a new Document with the given parameters and starts tracking it.

        Parameters
        ----------
        tokens: iterable of str
            The individual content tokens in the given document; will be fed directly
            into the various topic modelling algorithms. Assumed to have already
            undergone any necessary munging.
        metadata: dict, optional
            A general-purpose dictionary containing any metadata the user wants to
            track.
        display_str: str, optional
            The content of the document, as a single string containing any necessary
            markup or other formatting for human-readable display. By default,
            `display_str` is assumed to contain a HTML representation of the document
            (e.g., when the document is rendered in `ignis.aurum.nb_explore_topics()`),
            but a custom display function can be passed where necessary.
            If None, will use the Document tokens joined with single spaces.
        plain_text: str, optional
            The full text of the given document as a single normalised string. If
            `plain_text` is None, `display_str` is assumed to contain a HTML
            representation of the document, and a corresponding plain-text
            representation is automatically generated via BeautifulSoup when the
            attribute is first accessed.

        Returns
        -------
        str
            The ID for the added Document
        """
        if len(tokens) == 0:
            raise RuntimeError("Cannot add a Document with no tokens to a Corpus.")

        if metadata is None:
            metadata = collections.OrderedDict()
        if display_str is None:
            display_str = " ".join(tokens)
        doc = Document(tokens, metadata, display_str, plain_text)
        if doc.id in self.documents:
            raise RuntimeError(
                f"This Document's hash is already present in the Corpus; it may be a "
                f"duplicate. Ignoring.\n"
                f"(If this is a genuine hash collision, create a new Document with "
                f"different metadata values and try adding it again.)\n"
                f"{doc.id}\n"
                f"{doc.tokens}{doc.metadata}{doc.display_str}"
            )
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
        Mapping of IDs to Documents.  Ordered by Document ID as a form of shuffling
        (useful for things like preventing time bias in the document order).
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
        doc_id: str or uuid.UUID

        Returns
        -------
        Document
        """
        if isinstance(doc_id, str):
            doc_id = uuid.UUID(doc_id)
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

    def slice_by_tokens(self, tokens, include_root=False, plain_text=False):
        """
        Create a new CorpusSlice with Documents that contain at least one of the
        given tokens.

        If `plain_text` is True, will match `tokens` against the plain-text
        representation of the Document rather than its raw tokens (which might have
        undergone various transformations).

        `tokens` is canonically an iterable of single tokens, but if `plain_text` is
        set, a search will be done using the literal text of each element, allowing
        for exact phrase matching as well.

        If `include_root` is True, will also search the root Corpus for Documents
        instead of limiting the search to the current CorpusSlice.

        Parameters
        ----------
        tokens: iterable of str
            A list of the tokens to search Documents for
        include_root: bool, optional
            Whether or not to search the root Corpus as well
        plain_text: bool, optional
            Whether or not to search the plain-text representation of the Document
            rather than its tokenised form

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

        # By-token search matches tokens directly
        search_tokens = set(tokens)

        # Plain-text search performs a regex text search
        search_patterns = [
            re.compile(fr"(\s|^){re.escape(token)}(\s|$)") for token in tokens
        ]

        found_doc_ids = []
        for doc_id, doc in search_docs.items():
            if plain_text:
                doc_text = doc.plain_text

                found_pattern = False
                for pattern in search_patterns:
                    if pattern.search(doc_text):
                        found_pattern = True
                        break

                if found_pattern:
                    found_doc_ids.append(doc_id)

            else:
                doc_tokens = set(doc.tokens)
                if len(search_tokens & doc_tokens) > 0:
                    found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def slice_without_tokens(self, tokens, include_root=False, plain_text=False):
        """
        Returns a new CorpusSlice with the Documents that contain `tokens` removed.

        If `plain_text` is True, will match `tokens` against the plain-text
        representation of the Document rather than its raw tokens (which might have
        undergone various transformations).

        `tokens` is canonically an iterable of single tokens, but if `plain_text` is
        set, a search will be done using the literal text of each element, allowing
        for exact phrase matching as well.

        If `include_root` is True, will also search the root Corpus for Documents
        instead of limiting the search to the current CorpusSlice.

        Parameters
        ----------
        tokens: iterable of str
            The tokens (or phrases) to remove
        include_root: bool, optional
            Whether or not to search the root Corpus as well
        plain_text: bool, optional
            Whether or not to search the plain-text representation of the Document
            rather than its tokenised form

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

        # By-token search matches tokens directly
        search_tokens = set(tokens)

        # Plain-text search performs a regex text search
        search_patterns = [
            re.compile(fr"(\s|^){re.escape(token)}(\s|$)") for token in tokens
        ]

        filtered_doc_ids = []
        for doc_id, doc in search_docs.items():
            if plain_text:
                doc_text = doc.plain_text

                found_pattern = False
                for pattern in search_patterns:
                    if pattern.search(doc_text):
                        found_pattern = True
                        break

                if not found_pattern:
                    filtered_doc_ids.append(doc_id)

            else:
                doc_tokens = set(doc.tokens)
                if len(search_tokens & doc_tokens) == 0:
                    filtered_doc_ids.append(doc_id)

        return self.slice_by_ids(filtered_doc_ids)

    def slice_filter(self, filter_fn, include_root=False):
        """
        Returns a new CorpusSlice with the Documents that `filter_fn` returns True for.

        `filter_fn` receives one argument, a single Document in this CorpusSlice.

        Parameters
        ----------
        filter_fn: fn
            The filter function
        include_root: bool, optional
            Whether or not to search the root Corpus as well

        Returns
        -------
        CorpusSlice
        """
        if include_root:
            search_docs = self.root.documents
        else:
            search_docs = self.documents

        filtered_doc_ids = []
        for doc_id, doc in search_docs.items():
            if filter_fn(doc):
                filtered_doc_ids.append(doc_id)

        return self.slice_by_ids(filtered_doc_ids)

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
            if not isinstance(other_slice, CorpusSlice):
                raise RuntimeError(
                    "CorpusSlices can only be concatenated with other CorpusSlices."
                )

            if other_slice.root != self.root:
                raise RuntimeError(
                    "CorpusSlices can only be concatenated if they have the same root "
                    "Corpus."
                )

            slice_ids = set(other_slice.documents)
            new_slice_ids = new_slice_ids | slice_ids

        new_slice_ids = list(new_slice_ids)

        return CorpusSlice(self.root, new_slice_ids)

    def __add__(self, other):
        return self.concat(other)

    def __eq__(self, other):
        return (
            isinstance(other, CorpusSlice)
            and self.root == other.root
            and self.documents == other.documents
        )

    def nb_explore(self, doc_sort_key=None, reverse=False, display_fn=None):
        """
        Convenience function that creates an interactive Jupyter notebook widget for
        exploring the Documents contained in this CorpusSlice.

        Documents do not have any sort order imposed on them by default, but a custom
        sorting function can be passed via `doc_sort_key` as well.

        Parameters
        ----------
        doc_sort_key: fn, optional
            If specified, will sort documents using this key when displaying them.
        reverse: bool, optional
            Sets the sort direction for `doc_sort_key`, if specified.
        display_fn: fn, optional
            Custom display function that receives an individual Document as input and
            should display the Document in human-readable form as a side effect.
            If unset, will assume that the human-readable representation of the
            Document is in HTML format and display it accordingly.

        Returns
        -------
        ipywidgets.interact function
        """
        import ipywidgets
        from IPython.core.display import display, HTML

        docs = list(self.documents.values())

        if doc_sort_key is not None:
            docs = sorted(docs, key=doc_sort_key, reverse=reverse)

        def show_doc(index=0):
            print(f"[Total documents: {len(docs)}]\n")
            doc = docs[index]

            if display_fn is None:
                # Default HTML display
                print(f"ID: {doc.id}")
                if "filename" in doc.metadata:
                    print(f"Filename: {doc.metadata['filename']}")

                # Jupyter notebooks will interpret anything between $ signs
                # as LaTeX formulae when rendering HTML output, so we need to
                # replace them with escaped $ signs (only in Jupyter
                # environments)
                display_str = doc.display_str.replace("$", r"\$")

                # noinspection PyTypeChecker
                display(HTML(display_str))
            else:
                # User-provided display function
                display_fn(doc)

        return ipywidgets.interact(
            show_doc,
            index=ipywidgets.IntSlider(
                description="Document", min=0, max=len(docs) - 1
            ),
        )


class Document(object):
    """
    Documents hold the textual content of each file in the Corpus, as well as any
    relevant metadata.

    Parameters
    ----------
    tokens: iterable of str
        The individual content tokens in the given document; will be fed directly
        into the various topic modelling algorithms. Assumed to have already
        undergone any necessary munging.
    metadata: dict
        A general-purpose dictionary containing any metadata the user wants to
        track.
    display_str: str
        The content of the document, as a single string containing any necessary
        markup or other formatting for human-readable display. By default,
        `display_str` is assumed to contain a HTML representation of the document (
        e.g., when the document is rendered in `ignis.aurum.nb_explore_topics()`),
        but a custom display function can be passed where necessary.
    plain_text: str, optional
        The full text of the given document as a single normalised string. If
        `plain_text` is None, `display_str` is assumed to contain a HTML representation
        of the document, and a corresponding plain-text representation is
        automatically generated via BeautifulSoup when the attribute is first accessed.
    """

    # Let's make Document IDs deterministic on their data, so that multiple runs of a
    # Corpus creation script don't generate different IDs.
    # We will create a UUID5 for each Document against this fixed namespace:
    ignis_uuid_namespace = uuid.UUID("58ca78f2-0347-4b96-b2e7-63796bf87889")

    def __init__(self, tokens, metadata, display_str, plain_text=None):
        self.tokens = tokens
        self.metadata = metadata
        self.display_str = display_str
        self.plain_text = plain_text

        data = f"{tokens}{metadata}{display_str}"
        self.id = uuid.uuid5(Document.ignis_uuid_namespace, data)

    def __str__(self):
        metadata = json.dumps(self.metadata, indent=2)

        truncated = []
        for line in metadata.splitlines():
            if len(line) > 120:
                truncated.append(f"{line[:120]}...")
            else:
                truncated.append(line)
        metadata = "\n".join(truncated)

        return f"ID: {self.id}\n\nMetadata: {metadata}\n\n" f"{self.display_str}"

    def __getattribute__(self, item):
        if item == "plain_text" and object.__getattribute__(self, "plain_text") is None:
            # There is no `plain_text` set for this document; assume that
            # `display_str` contains a HTML representation of the document.
            soup = BeautifulSoup(self.display_str, "lxml")

            # The text returned by BeautifulSoup might contain whitespace --
            # Concatenate, split, and concatenate again to normalise the spacing
            self.plain_text = " ".join(soup.get_text().split())
            return self.plain_text
        return object.__getattribute__(self, item)


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
