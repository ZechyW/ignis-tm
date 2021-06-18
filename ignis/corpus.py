"""
`ignis.corpus.Corpus` and `ignis.corpus.CorpusSlice` instances are containers for
tracking the `ignis.corpus.Document` objects in a dataset.
"""

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
    Conceptually, `Corpus` objects contain the full amount of data for a given dataset.

    Some subset of the `Corpus` (up to the full `Corpus` itself) must be sliced into a
    `CorpusSlice` to perform topic modelling over the data, and these `CorpusSlice`
    objects can be iteratively expanded or contracted freely within the full set
    of `Corpus` data.

    The same `Corpus` will be used even as sub-slices of the data go through iterative
    modelling -- Smaller sets of `Document` objects will just be selected by ID.

    Note: `Document` objects in a `Corpus` are loosely kept in insertion order,
    but they are shuffled when sliced into `CorpusSlice` objects (viz., they are
    sorted by their randomly-generated IDs).

    Stop words are also managed at the (root) `Corpus` level -- Whenever `Document`
    objects are retrieved via `Corpus.get_document()`, stop words are removed from
    their tokens at run-time.

    Parameters
    ----------
    stop_words: set of str, optional
        The initial list of corpus-level stop words, if any.

    Attributes
    ----------
    root: Corpus
        A reference to this base `Corpus` for bookkeeping so that slicing can be done
        with both `Corpus` and `CorpusSlice` instances.
    """

    def __init__(self, stop_words=None):
        self.root = self
        if stop_words is not None:
            self._stop_words = set(stop_words)
        else:
            self._stop_words = set()
        self._documents = collections.OrderedDict()

    # --------------------
    # Data manipulation
    def add_doc(self, tokens, metadata=None, display_str=None, plain_text=None):
        """
        Creates a new `Document` with the given parameters and starts tracking it.

        Parameters
        ----------
        tokens: iterable of str
            The individual content tokens in the given document; will be fed directly
            into the various topic modelling algorithms.

            Assumed to have already undergone any necessary pre-processing.
        metadata: dict, optional
            A general-purpose dictionary containing any metadata the user wants to
            track.
        display_str: str, optional
            The content of the document, as a single string containing any necessary
            markup or other formatting for human-readable display. By default,
            `display_str` is assumed to contain a HTML representation of the document
            (e.g., when the document is rendered via
            `ignis.aurum.Aurum.nb_explore_topics()`), but a custom display function
            can be passed where necessary.

            If `None`, will use the document's tokens joined with single spaces.
        plain_text: str, optional
            The full text of the given document as a single normalised string.

            If `plain_text` is `None`, `display_str` is assumed to contain a HTML
            representation of the document, and a corresponding plain-text
            representation is automatically generated via `BeautifulSoup` when the
            attribute is first accessed.

        Returns
        -------
        uuid.UUID
            The ID for the added `Document`.
        """
        if len(tokens) == 0:
            raise RuntimeError("Cannot add a Document with no tokens to a Corpus.")

        if metadata is None:
            metadata = collections.OrderedDict()
        if display_str is None:
            display_str = " ".join(tokens)
        doc = Document(tokens, metadata, display_str, plain_text)
        if doc.id in self._documents:
            raise RuntimeError(
                f"This Document's hash is already present in the Corpus; it may be a "
                f"duplicate. Ignoring.\n"
                f"(If this is a genuine hash collision, create a new Document with "
                f"different metadata values and try adding it again.)\n"
                f"{doc.id}\n"
                f"{doc.tokens}{doc.metadata}{doc.display_str}"
            )
        self._documents[doc.id] = doc

        # Impose Corpus stop word list
        # We pass the actual private `_stop_words` set by reference so that
        # modifications to the root corpus stop word list will automatically
        # propagate to all children Documents and CorpusSlices.
        self._documents[doc.id].stop_words = self.root._stop_words

        return doc.id

    def save(self, filename):
        """
        Saves the `Corpus` or `CorpusSlice` object to the given file.
        Essentially uses a bz2-compressed Pickle format.

        We recommend using _.corpus_ as the canonical file extension for `Corpus`
        objects and _.slice_ as the canonical file extension for `CorpusSlice` objects,
        but this is not strictly enforced by the library.

        Parameters
        ----------
        filename: str or pathlib.Path
            File to save the `Corpus` or `CorpusSlice` to.
        """
        filename = pathlib.Path(filename)
        with bz2.open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def concat(self, *other_corpora):
        """
        Consolidates two or more separate `Corpus` objects into a single `Corpus`.

        Functionally different from the `CorpusSlice.concat()` method, which
        consolidates two slices that necessarily come from the same single `Corpus`.

        No ID collision checks are done as part of the concatenation, but these
        should be highly improbable to happen with UUIDs.

        The stop word lists from all the concatenated `Corpus` objects are combined
        to form the stop word list for the final `Corpus`.

        Note: The `+` operator can also be used to concatenate `Corpus` objects.

        Parameters
        ----------
        *other_corpora: iterable of Corpus
            The other `Corpus` object(s) to include.

        Returns
        -------
        Corpus
        """
        # Sanity check
        for other_corpus in other_corpora:
            if not type(other_corpus) is Corpus:
                raise RuntimeError(
                    "Corpus objects can only be concatenated with other Corpus "
                    "objects."
                )

        # Consolidate
        new_corpus = Corpus()

        # Because we are touching the private `._documents` store directly, we also
        # need to manually handle stop word lists.
        new_corpus.add_stop_words(self.stop_words)
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            new_corpus._documents[doc.id] = doc
            new_corpus._documents[doc.id].stop_words = new_corpus.root._stop_words

        for other_corpus in other_corpora:
            new_corpus.add_stop_words(other_corpus.stop_words)
            for doc_id in other_corpus.document_ids:
                doc = other_corpus.get_document(doc_id)
                new_corpus._documents[doc.id] = doc
                new_corpus._documents[doc.id].stop_words = new_corpus.root._stop_words

        # Return
        return new_corpus

    def add_stop_word(self, stop_word):
        """
        Adds the given stop word to the root `Corpus` object's stop word list.

        Parameters
        ----------
        stop_word: str
        """
        if not isinstance(stop_word, str):
            raise RuntimeWarning(
                "Use `.add_stop_words()` to add multiple stop words at once."
            )

        self.root.add_stop_words([stop_word])

    def add_stop_words(self, stop_words):
        """
        Adds the given stop words to the root `Corpus` object's stop word list.

        Parameters
        ----------
        stop_words: iterable of str
        """
        if isinstance(stop_words, str):
            raise RuntimeWarning("Use `.add_stop_word()` to add a single stop word.")

        for stop_word in stop_words:
            self.root._stop_words.add(stop_word)

    def remove_stop_word(self, stop_word):
        """
        Remove the given stop word from the root `Corpus` object's stop word list.

        Parameters
        ----------
        stop_word: str
        """
        if not isinstance(stop_word, str):
            raise RuntimeWarning(
                "Use `.remove_stop_words()` to remove multiple stop words at once."
            )

        self.root.remove_stop_words([stop_word])

    def remove_stop_words(self, stop_words):
        """
        Remove the given stop words from the root `Corpus` object's stop word list.

        Parameters
        ----------
        stop_words: iterable of str
        """
        if isinstance(stop_words, str):
            raise RuntimeWarning(
                "Use `.remove_stop_word()` to remove a single stop word."
            )

        for stop_word in stop_words:
            self.root._stop_words.remove(stop_word)

    @property
    def stop_words(self):
        """
        Return a copy of the set of stop words defined for this root `Corpus`.

        Returns
        -------
        set of str
        """
        return set(self.root._stop_words)

    def __add__(self, other):
        return self.concat(other)

    def __len__(self):
        return len(self._documents)

    @property
    def document_ids(self):
        """
        Get a list of the `Document` IDs tracked by this `Corpus`.

        Returns
        -------
        iterable of uuid.UUID
        """
        return list(self._documents.keys())

    def get_document(self, doc_id):
        """
        Return the `Document` from this `Corpus` with the given ID.

        Parameters
        ----------
        doc_id: uuid.UUID or str

        Returns
        -------
        Document
        """
        if isinstance(doc_id, str):
            doc_id = uuid.UUID(doc_id)

        return self._documents[doc_id]

    # --------------------
    # Indexing and iteration
    def __getitem__(self, doc_id):
        if isinstance(doc_id, int):
            # Pick out the n-th Document in the Corpus
            return self.get_document(self.document_ids[doc_id])
        elif isinstance(doc_id, slice):
            # Pick out a slice of Documents from the Corpus
            return [
                self.get_document(slice_doc) for slice_doc in self.document_ids[doc_id]
            ]
        else:
            # Pick out the Document by ID
            return self.get_document(doc_id)

    def __iter__(self):
        for doc_id in self.document_ids:
            yield self.get_document(doc_id)

    # --------------------
    # Slicing functions
    def slice_full(self):
        """
        Get a `CorpusSlice` containing all the `Document` objects in this `Corpus`.

        **Note**: This method can only be used with base `Corpus` objects, and will
        raise an error if used with a `CorpusSlice` object.

        Returns
        -------
        CorpusSlice
        """
        return CorpusSlice(self, self.document_ids)

    def slice_by_ids(self, doc_ids):
        """
        Create a new `CorpusSlice` with the given `Document` IDs.

        Parameters
        ----------
        doc_ids: iterable of uuid.UUID or iterable of str
            List of `Document` IDs to include.

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if isinstance(doc_ids, str) or isinstance(doc_ids, uuid.UUID):
            raise RuntimeWarning(
                "`.slice_by_ids()` expects an iterable of Document IDs. If you are "
                "certain you want to create a slice containing a single Document, "
                "wrap it in a list or tuple before passing it as input."
            )

        # Make sure the IDs are in this instance's documents.
        for doc_id in doc_ids:
            if isinstance(doc_id, str):
                doc_id = uuid.UUID(doc_id)
            if doc_id not in self._documents:
                raise RuntimeError(
                    f"There is no `Document` with ID '{str(doc_id)}' within the "
                    f"current slice."
                )

        return CorpusSlice(self.root, doc_ids)

    def slice_by_token(self, token):
        """
        Create a new `CorpusSlice` with `Document` objects that contain the given
        token.

        This method searches the `.tokens` property of each `Document`
        (case-insensitive). These are the words and phrases that appear as labels in
        the main graphical visualisation via `ignis.aurum.Aurum.show_visualisation()`.

        Note that each label is a single token even if it contains spaces/multiple
        words.

        Parameters
        ----------
        token: str
            The token to search `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        if not isinstance(token, str):
            raise RuntimeWarning(
                "Use `.slice_by_tokens()` to slice by multiple tokens at once."
            )

        return self.slice_by_tokens([token])

    def slice_by_tokens(self, tokens):
        """
        Create a new `CorpusSlice` with `Document` objects that contain at least one
        of the given tokens (an "OR" search).

        This method searches the `.tokens` property of each `Document`
        (case-insensitive). These are the words and phrases that appear as labels in
        the main graphical visualisation via `ignis.aurum.Aurum.show_visualisation()`.

        Note that each label is a single token even if it contains spaces/multiple
        words.

        To slice together `Document` objects that contain multiple specific tokens (an
        "AND" search), you can chain multiple invocations of the single token filter.
        E.g.:
        `corpus.slice_by_token("token").slice_by_token("token 2")`.

        Parameters
        ----------
        tokens: iterable of str
            A list of the tokens to search `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if isinstance(tokens, str):
            raise RuntimeWarning("Use `.slice_by_token()` to slice by a single token.")

        # By-token search matches tokens directly, ignoring case
        search_tokens = set([token.lower() for token in tokens])

        found_doc_ids = []
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            doc_tokens = set([token.lower() for token in doc.tokens])
            if len(search_tokens & doc_tokens) > 0:
                found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def slice_by_text_string(self, text_string):
        """
        Create a new `CorpusSlice` with `Document` objects that contain the given
        text string as an exact phrase match.

        This method searches the `.plain_text` property of each `Document`
        (case-insensitive). This is the full human-readable representation of the
        `Document` as shown in the `ignis.aurum.Aurum.nb_explore_topics()` or
        `Corpus.nb_explore()` widgets.

        Non-word characters at the boundaries of the text string are ignored, but any
        punctuation marks or other characters *within* the `Document` text have to be
        in the search string as well.

        For example, the multi-word text string `"test this"` will match against a
        `Document` with the phrase `"She said, 'Test this!'"`, but will *not* match
        against a `Document` with the phrase `"A test, this is."`, because of the
        intervening comma.

        To match both documents, you can perform an "OR" search on the individual
        words using `Corpus.slice_by_text_strings()` instead:
        `corpus_slice.slice_by_text_strings(["test", "this"])`.

        Parameters
        ----------
        text_string: str
            The text string to search the content of `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        if not isinstance(text_string, str):
            raise RuntimeWarning(
                "Use `.slice_by_text_strings()` to slice by multiple text strings."
            )

        return self.slice_by_text_strings([text_string])

    def slice_by_text_strings(self, text_strings):
        """
        Create a new `CorpusSlice` with `Document` objects that contain at least one of
        the given text strings (an "OR" search).

        This method searches the `.plain_text` property of each `Document`
        (case-insensitive). This is the full human-readable representation of the
        `Document` as shown in the `ignis.aurum.Aurum.nb_explore_topics()` or
        `Corpus.nb_explore()` widgets.

        Non-word characters at the boundaries of each text string are ignored.

        To slice together `Document` objects that contain multiple specific text
        strings (an "AND" search), you can chain multiple invocations of the single
        text string filter.
        E.g.:
        `corpus.slice_by_text_string("exact phrase").slice_by_text_string("another
        exact phrase")`.

        Parameters
        ----------
        text_strings: iterable of str
            A list of the text strings to search the content of `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if isinstance(text_strings, str):
            raise RuntimeWarning(
                "Use `.slice_by_text_string()` to slice by a single text string."
            )

        # Plain-text search performs a regex text search, looking for `text_string`
        # matches that start/end on word boundaries.
        search_patterns = [
            re.compile(fr"(\b|\s|^){re.escape(text_string)}(\b|\s|$)", re.IGNORECASE)
            for text_string in text_strings
        ]

        found_doc_ids = []
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            doc_text = doc.plain_text

            found_pattern = False
            for pattern in search_patterns:
                if pattern.search(doc_text):
                    found_pattern = True
                    break

            if found_pattern:
                found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def slice_without_token(self, token):
        """
        Create a new `CorpusSlice` by removing `Document` objects that contain the given
        token.

        This method searches the `.tokens` property of each `Document`
        (case-insensitive). These are the words and phrases that appear as labels in
        the main graphical visualisation via `ignis.aurum.Aurum.show_visualisation()`.

        Note that each label is a single token even if it contains spaces/multiple
        words.

        Parameters
        ----------
        token: str
            The token to search `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        if not isinstance(token, str):
            raise RuntimeWarning(
                "Use `.slice_without_tokens()` to slice using multiple tokens at once."
            )

        return self.slice_without_tokens([token])

    def slice_without_tokens(self, tokens):
        """
        Create a new `CorpusSlice` by removing `Document` objects that contain any of
        the given tokens (an "OR" search).

        This method searches the `.tokens` property of each `Document`
        (case-insensitive). These are the words and phrases that appear as labels in
        the main graphical visualisation via `ignis.aurum.Aurum.show_visualisation()`.

        Note that each label is a single token even if it contains spaces/multiple
        words.

        To slice out `Document` objects that contain multiple specific tokens (an "AND"
        search), you can chain multiple invocations of the single token filter.
        E.g.:
        `corpus.slice_without_token("token").slice_without_token("token 2")`.

        Parameters
        ----------
        tokens: iterable of str
            A list of the tokens to search `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if isinstance(tokens, str):
            raise RuntimeWarning(
                "Use `.slice_without_token()` to slice using a single token."
            )

        # By-token search matches tokens directly, ignoring case
        search_tokens = set([token.lower() for token in tokens])

        found_doc_ids = []
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            doc_tokens = set([token.lower() for token in doc.tokens])
            if len(search_tokens & doc_tokens) == 0:
                found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def slice_without_text_string(self, text_string):
        """
        Create a new `CorpusSlice` with `Document` objects that contain the given
        text string removed.

        This method searches the `.plain_text` property of each `Document`
        (case-insensitive). This is the full human-readable representation of the
        `Document` as shown in the `ignis.aurum.Aurum.nb_explore_topics()` or
        `Corpus.nb_explore()` widgets.

        Non-word characters at the boundaries of the text string are ignored, but any
        punctuation marks or other characters *within* the `Document` text have to be
        in the search string as well.

        For example, the multi-word text string `"test this"` will match against a
        `Document` with the phrase `"She said, 'Test this!'"`, but will *not* match
        against a `Document` with the phrase `"A test, this is."`, because of the
        intervening comma.

        To match both documents, you can perform an "OR" search on the individual
        words using `Corpus.slice_without_text_strings()` instead:
        `corpus_slice.slice_without_text_strings(["test", "this"])`.

        Parameters
        ----------
        text_string: str
            The text string to search the content of `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        if not isinstance(text_string, str):
            raise RuntimeWarning(
                "Use `.slice_without_text_strings()` to slice using multiple text "
                "strings at once."
            )

        return self.slice_without_text_strings([text_string])

    def slice_without_text_strings(self, text_strings):
        """
        Create a new `CorpusSlice` by removing `Document` objects that contain any of
        the given text strings (an "OR" search).

        This method searches the `.plain_text` property of each `Document`
        (case-insensitive). This is the full human-readable representation of the
        `Document` as shown in the `ignis.aurum.Aurum.nb_explore_topics()` or
        `Corpus.nb_explore()` widgets.

        Non-word characters at the boundaries of each text string are ignored.

        To slice out `Document` objects that contain multiple specific text strings
        (an "AND" search), you can chain multiple invocations of the single text string
        filter.
        E.g.:
        `corpus.slice_without_text_string("exact phrase").slice_without_text_string(
        "another exact phrase")`.

        Parameters
        ----------
        text_strings: iterable of str
            A list of the text strings to search the content of `Document` objects for.

        Returns
        -------
        CorpusSlice
        """
        # Sanity check
        if isinstance(text_strings, str):
            raise RuntimeWarning(
                "Use `.slice_without_text_string()` to slice using a single text "
                "string."
            )

        # Plain-text search performs a regex text search, looking for `text_string`
        # matches that start/end on word boundaries or whitespace.
        search_patterns = [
            re.compile(fr"(\b|\s|^){re.escape(text_string)}(\b|\s|$)", re.IGNORECASE)
            for text_string in text_strings
        ]

        found_doc_ids = []
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            doc_text = doc.plain_text

            found_pattern = False
            for pattern in search_patterns:
                if pattern.search(doc_text):
                    found_pattern = True
                    break

            if not found_pattern:
                found_doc_ids.append(doc_id)

        return self.slice_by_ids(found_doc_ids)

    def slice_filter(self, filter_fn):
        """
        Slices using a custom `filter_fn` that receives one argument, a single
        `Document`.

        Returns a new `CorpusSlice` with the `Document` objects that `filter_fn`
        returns `True` for.

        Parameters
        ----------
        filter_fn: function
            The filter function.

        Returns
        -------
        CorpusSlice
        """
        filtered_doc_ids = []
        for doc_id in self.document_ids:
            doc = self.get_document(doc_id)
            if filter_fn(doc):
                filtered_doc_ids.append(doc_id)

        return self.slice_by_ids(filtered_doc_ids)

    def nb_explore(
        self,
        doc_sort_key=None,
        reverse=False,
        display_fn=None,
        max_display_length=50000,
        metadata_full_doc_link="filename",
    ):
        """
        Convenience function that creates an interactive Jupyter notebook widget for
        exploring the `Document` objects tracked by this `Corpus` or `CorpusSlice`.

        `Document` objects do not have any sort order imposed on them by default,
        but a custom sorting function can be passed via `doc_sort_key` if necessary.

        Parameters
        ----------
        doc_sort_key: fn, optional
            If specified, will sort `Document` objects using this key when displaying
            them.
        reverse: bool, optional
            Reverses the sort direction for `doc_sort_key`, if specified.
        display_fn: fn, optional
            Custom display function that receives an individual `Document` as input,
            and should display the `Document` in human-readable form as a side effect.

            If unset, will assume that the human-readable representation of the
            `Document` is in HTML format and display it accordingly.
        max_display_length: int, optional
            Maximum length (in characters) to display from the `Document` object's
            human-readable representation.  If the `Document` object's human-readable
            representation is longer than this limit, a link will be generated to
            view the full human-readable representation in a new window.

            No effect if `display_fn` is set.
        metadata_full_doc_link: str, optional
            If `max_display_length` is exceeded, this key is used to get the path to
            the full human-readable representation from the `Document` object's
            metadata dictionary.

            No effect if `display_fn` is set.
        """
        import ipywidgets
        from IPython.core.display import display, HTML
        import ignis.util.jupyter_styles

        # Set up widget styling
        # noinspection PyTypeChecker
        display(HTML(ignis.util.jupyter_styles.jupyter_output_style))

        if len(self) == 0:
            print("This Corpus or CorpusSlice contains no documents.")
            return

        docs = list(self)

        if doc_sort_key is not None:
            docs = sorted(docs, key=doc_sort_key, reverse=reverse)

        def show_doc(index=1):
            # Start `index` from 1 for user-friendliness
            print(f"[Total documents: {len(docs)}]\n")
            doc = docs[index - 1]

            if display_fn is None:
                # Default HTML display
                print(f"ID: {doc.id}")
                if metadata_full_doc_link in doc.metadata:
                    print(f"Full document: {doc.metadata[metadata_full_doc_link]}")

                # Jupyter notebooks will interpret anything between $ signs as LaTeX
                # formulae when rendering HTML output, so we need to replace them
                # with escaped $ signs (only in Jupyter environments)
                display_str = doc.display_str.replace("$", r"\$")

                # Length check
                if len(display_str) > max_display_length:
                    display_str = (
                        f"<p>"
                        f"<b>Document too long to display in full - "
                        f"Showing first {max_display_length} characters.</b>"
                        f"</p>"
                        f"<p>"
                        f"<a href="
                        f"'{doc.metadata[metadata_full_doc_link]}' "
                        f"target='_blank'>"
                        f"Click here"
                        f"</a> to open the full document in a new tab/window."
                        f"</p>"
                        f"<hr/>" + display_str[:max_display_length]
                    )

                # noinspection PyTypeChecker
                display(HTML(display_str))
            else:
                # User-provided display function
                display_fn(doc)

        # Control and output widgets for the document viewer
        slider = ipywidgets.IntSlider(
            description="Document",
            min=1,
            max=len(docs),
            continuous_update=False,
            layout=ignis.util.jupyter_styles.slider_layout,
            style=ignis.util.jupyter_styles.slider_style,
        )
        text = ipywidgets.BoundedIntText(
            min=1, max=len(docs), layout=ignis.util.jupyter_styles.slider_text_layout
        )
        ipywidgets.jslink((slider, "value"), (text, "value"))
        ui = ipywidgets.HBox([slider, text])
        out = ipywidgets.interactive_output(show_doc, {"index": slider})
        # noinspection PyTypeChecker
        display(ui, out)


class CorpusSlice(Corpus):
    """
    Contains some subset of the `Document` objects in a `Corpus`, and keeps a reference
    to the root `Corpus` for bookkeeping and iteration.

    All the `ignis` topic models take `CorpusSlice` objects as input.

    `CorpusSlice` is a subclass of `Corpus`, so the same slicing methods can be used
    with instances of both classes. To restart the slicing process from the full
    base `Corpus`, you can use the `.root` property of a `CorpusSlice` object.
    (E.g., `corpus_slice.root.<method>()`.)

    Parameters
    ----------
    root: Corpus
        The root `Corpus` instance for this slice.
    slice_ids: iterable of uuid.UUID or iterable of str
        The IDs for the `Document` objects to include in this slice.

        These are canonically instances of `uuid.UUID`, but a list of strings can
        also be passed (e.g., when instantiating `CorpusSlice` objects interactively).

    Attributes
    ----------
    root: Corpus
        A reference to the base `Corpus` for bookkeeping and slicing/iteration.
    """

    def __init__(self, root, slice_ids):
        super().__init__()

        self.root = root
        self._documents = collections.OrderedDict()

        slice_ids.sort()
        for slice_id in slice_ids:
            if isinstance(slice_id, str):
                slice_id = uuid.UUID(slice_id)
            self._documents[slice_id] = root.get_document(slice_id)

    def slice_full(self):
        # This method can only be used with base `Corpus` objects, and not
        # `CorpusSlice` objects.
        raise RuntimeError(
            "The `.slice_full()` method can only be used with `Corpus` objects, "
            "not `CorpusSlice` objects."
        )

    def concat(self, *other_slices):
        """
        Returns a new `CorpusSlice` that has the `Document` objects from this instance
        and all the other specified `CorpusSlice` objects combined.

        Only `CorpusSlice` objects with the same root `Corpus` can be concatenated.

        Note: The `+` operator can also be used to concatenate `CorpusSlice` objects.

        Parameters
        ----------
        *other_slices: iterable of CorpusSlice
            The other `CorpusSlice` object(s) to include.

        Returns
        -------
        CorpusSlice
        """
        new_slice_ids = set(self.document_ids)

        for other_slice in other_slices:
            if not type(other_slice) is CorpusSlice:
                raise RuntimeError(
                    "CorpusSlices can only be concatenated with other CorpusSlices."
                )

            if other_slice.root != self.root:
                raise RuntimeError(
                    "CorpusSlices can only be concatenated if they have the same root "
                    "Corpus."
                )

            slice_ids = set(other_slice.document_ids)
            new_slice_ids = new_slice_ids | slice_ids

        new_slice_ids = list(new_slice_ids)

        return CorpusSlice(self.root, new_slice_ids)

    def __add__(self, other):
        return self.concat(other)

    def __eq__(self, other):
        return (
            type(other) is CorpusSlice
            and self.root == other.root
            and set(self.document_ids) == set(other.document_ids)
        )


class Document(object):
    """
    `Document` objects hold the textual content of each entry in the `Corpus`, as well
    as any relevant metadata.

    Parameters
    ----------
    tokens: iterable of str
        The individual content tokens in the given document, which will be fed
        directly into the various topic modelling algorithms.

        Assumed to have already undergone any necessary pre-processing except
        stop word removal, which will be done at run-time whenever `Document.tokens` is
        accessed.
    metadata: dict
        A general-purpose dictionary containing any metadata the user wants to
        track.
    display_str: str
        The content of the document, as a single string containing any necessary
        markup or other formatting for human-readable display. By default,
        `display_str` is assumed to contain a HTML representation of the document
        (e.g., when the document is rendered in
        `ignis.aurum.Aurum.nb_explore_topics()`), but a custom display function can
        be passed where necessary.
    plain_text: str, optional
        The full text of the given document as a single normalised string.

        If `plain_text` is None, `display_str` is assumed to contain a HTML
        representation of the document, and a corresponding plain-text representation is
        automatically generated via BeautifulSoup when the attribute is first accessed.

    Attributes
    ----------
    stop_words: set of str
        The set of current stop words for this `Document` object. When a `Document` is
        added to a `Corpus` via `Corpus.add_doc()`, this becomes a reference to the
        root `Corpus` object's list of stop words.
        Any items in this set of stop words will be removed when this `Document`
        object's `tokens` property is accessed.
    """

    # Let's make Document IDs deterministic on their data, so that multiple runs of a
    # Corpus creation script don't generate different IDs.
    # We will create a UUID5 for each Document against this fixed namespace:
    ignis_uuid_namespace = uuid.UUID("58ca78f2-0347-4b96-b2e7-63796bf87889")
    """The UUID5 namespace for generating deterministic `Document` IDs."""

    def __init__(self, tokens, metadata, display_str, plain_text=None):
        self.raw_tokens = tokens
        self.metadata = metadata
        self.display_str = display_str
        self.plain_text = plain_text
        self.stop_words = set()

        data = f"{tokens}{metadata}{display_str}"
        self.id = uuid.uuid5(Document.ignis_uuid_namespace, data)

    def __setstate__(self, state):
        # Called when unpickling Document objects.
        # Ensure that `raw_tokens` is set properly when loading Corpus files saved
        # using the previous format (< v1.5.0)
        if "tokens" in state and "raw_tokens" not in state:
            state["raw_tokens"] = state["tokens"]
        self.__dict__.update(state)

    @property
    def tokens(self):
        return [token for token in self.raw_tokens if token not in self.stop_words]

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

    def __repr__(self):
        return str(self)


# noinspection PyProtectedMember
def load_corpus(filename):
    """
    Loads a `ignis.corpus.Corpus` object from the given file.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the `ignis.corpus.Corpus` object from.

    Returns
    -------
    ignis.corpus.Corpus
    """
    with bz2.open(filename, "rb") as fp:
        loaded = pickle.load(fp)

    if not type(loaded) is Corpus:
        raise ValueError(f"File does not contain a `Corpus` object: '{filename}'")

    # Re-initialise the `Corpus` with all `Documents` and the dynamic stop word list.
    new_corpus = Corpus()
    if hasattr(loaded, "_stop_words"):
        # Copy stop words, if they are set
        new_corpus._stop_words = set(loaded._stop_words)
    if hasattr(loaded, "documents"):
        # Old version - `.documents` was directly accessible.
        docs = loaded.documents.values()
    else:
        # New version - documents can only be retrieved via `get_document()`
        docs = loaded._documents.values()
    for doc in docs:
        # `Document` objects are un-pickled separately; `Document.__setstate__()`
        # ensures that the `raw_tokens` attribute is set appropriately.
        new_corpus.add_doc(
            doc.raw_tokens, doc.metadata, doc.display_str, doc.plain_text
        )

    return new_corpus


def load_slice(filename):
    """
    Loads a `ignis.corpus.CorpusSlice` object from the given file.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the `ignis.corpus.CorpusSlice` object from.

    Returns
    -------
    ignis.corpus.CorpusSlice
    """
    with bz2.open(filename, "rb") as fp:
        loaded = pickle.load(fp)

    if not type(loaded) is CorpusSlice:
        raise ValueError(f"File does not contain a `CorpusSlice` object: '{filename}'")

    return loaded
