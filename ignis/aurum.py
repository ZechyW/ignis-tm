"""
`ignis.aurum.Aurum` instances manage the results of topic modelling runs, and provide
methods for exploring and iterating over them.
"""

import bz2
import copy
import json
import pathlib
import pickle
import pprint
import tempfile

import ignis.labeller.tomotopy
import ignis.vis.pyldavis


class Aurum:
    """
    `Aurum` objects bring together the public APIs for `ignis` models, automated
    labellers, and visualisation data providers, while also providing general
    save/load functionality.

    **NOTE**: All topic IDs retrieved from `Aurum` instances are 1-indexed rather than
    0-indexed. So a model with 5 topics has topic IDs `[1, 2, 3, 4, 5]` and not
    `[0, 1, 2, 3, 4]`.

    This is for easier matching against pyLDAvis visualisations, and for easier usage
    by non-technical users.

    Parameters
    ----------
    ignis_model: ignis.models.base.BaseModel
        The specific `ignis` model used to generate this `Aurum` object.
    """

    def __init__(self, ignis_model):
        self.ignis_model = ignis_model
        self.model_type = ignis_model.model_type
        self.model_options = ignis_model.options

        # Grab a reference to the CorpusSlice object so that we can use its methods
        # directly
        self.corpus_slice = ignis_model.corpus_slice

        # Aurum objects also optionally have cached labeller and visualisation data
        # objects
        self.labeller = None

        self.vis_type = None
        self.vis_options = None
        self.vis_data = None

    def save(self, filename):
        """
        Saves the `Aurum` object, including its associated `ignis.models` model,
        to the given file.
        Essentially uses a bz2-compressed Pickle format.

        Also attempts to save any cached visualisation data, but will omit any
        initialised automated labeller (since labellers are probably not pickle-able).

        We recommend using _.aurum_ as the canonical file extension, but this is not
        strictly enforced by the library.

        Parameters
        ----------
        filename: str or pathlib.Path
            File to save the model to.
        """
        filename = pathlib.Path(filename)

        # Copy the Ignis model, separate the external library's model out, pickle
        # everything together
        # (The model objects created by external libraries might not be pickle-able)
        external_model = self.ignis_model.model
        self.ignis_model.model = None
        save_model = copy.deepcopy(self.ignis_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "save_model.bin"
            # We assume that the external library's model object provides a `.save()`
            # method that takes the filename as a string
            tmp_model_file = str(tmp_model_file)
            external_model.save(tmp_model_file)
            with open(tmp_model_file, "rb") as fp:
                external_model_bytes = fp.read()

        save_object = {
            "save_model": save_model,
            "model_type": save_model.model_type,
            "external_model_bytes": external_model_bytes,
            # We should also be able to save any cached visualisation data, but the
            # labeller is probably not pickle-able.
            "vis_type": self.vis_type,
            "vis_options": self.vis_options,
            "vis_data": self.vis_data,
        }

        with bz2.open(filename, "wb") as fp:
            pickle.dump(save_object, fp)

        self.ignis_model.model = external_model

    # =================================================================================
    # Topic Model
    # (Default arguments are defined by the models themselves, in case different
    # values work better with different models)
    def get_num_topics(self):
        """
        See `ignis.models.base.BaseModel.get_num_topics()`.
        """
        return self.ignis_model.get_num_topics()

    def get_topic_words(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_topic_words()`.
        """
        return self.ignis_model.get_topic_words(*args, **kwargs)

    def get_topic_documents(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_topic_documents()`.
        """
        return self.ignis_model.get_topic_documents(*args, **kwargs)

    def get_document_topics(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_document_topics()`.
        """
        return self.ignis_model.get_document_topics(*args, **kwargs)

    def get_document_top_topic(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_document_top_topic()`.
        """
        return self.ignis_model.get_document_top_topic(*args, **kwargs)

    def get_coherence(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_coherence()`.
        """
        return self.ignis_model.get_coherence(*args, **kwargs)

    # =================================================================================
    # Corpus Slice
    @property
    def document_ids(self):
        """
        Get the IDs of all the `ignis.corpus.Document` objects that are covered by this
        `ignis` model.

        Returns
        -------
        iterable of uuid.UUID
        """
        return list(self.corpus_slice.document_ids)

    def get_document(self, *args, **kwargs):
        """
        See `ignis.corpus.CorpusSlice.get_document()`.
        """
        return self.corpus_slice.get_document(*args, **kwargs)

    def slice_by_ids(self, doc_ids, include_root=False):
        """
        Slice the model's input dataset by the given `ignis.corpus.Document` IDs.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_by_ids()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_by_ids(doc_ids)

    def slice_by_token(self, token, include_root=False):
        """
        Slice the model's input dataset by the given token.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_by_token()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_by_token(token)

    def slice_by_tokens(self, tokens, include_root=False):
        """
        Slice the model's input dataset by tokens.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_by_tokens()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_by_tokens(tokens)

    def slice_without_token(self, token, include_root=False):
        """
        Slice the model's input dataset by removing `ignis.corpus.Document` objects
        that contain the given token.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_without_token()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_without_token(token)

    def slice_without_tokens(self, tokens, include_root=False):
        """
        Slice the model's input dataset by removing `ignis.corpus.Document` objects
        that contain any of the given tokens.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_without_tokens()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_without_tokens(tokens)

    def slice_by_text_string(self, text_string, include_root=False):
        """
        Slice the model's input dataset by the given text string.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_by_text_string()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_by_text_string(text_string)

    def slice_by_text_strings(self, text_strings, include_root=False):
        """
        Slice the model's input dataset by the given text strings.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_by_text_strings()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_by_text_strings(text_strings)

    def slice_without_text_string(self, text_string, include_root=False):
        """
        Slice the model's input dataset by removing `ignis.corpus.Document` objects
        that contain the given text string.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_without_token()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_without_text_string(text_string)

    def slice_without_text_strings(self, text_strings, include_root=False):
        """
        Slice the model's input dataset by removing `ignis.corpus.Document` objects
        that contain any of the given text strings.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_without_tokens()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_without_text_strings(text_strings)

    def slice_filter(self, filter_fn, include_root=False):
        """
        Slice the model's input dataset using some custom `filter_fn`.

        If `include_root` is True, will slice the full base `ignis.corpus.Corpus`
        instead of just the model's current `ignis.corpus.CorpusSlice`.

        See `ignis.corpus.Corpus.slice_filter()` for more details.
        """
        base_slice = self.corpus_slice
        if include_root:
            base_slice = base_slice.root

        return base_slice.slice_filter(filter_fn)

    # =================================================================================
    # Automated Labeller
    def init_labeller(self, labeller_type, **labeller_options):
        """
        Trains an automated `ignis.labeller` for this `Aurum` object.

        Parameters
        ----------
        labeller_type: {"tomotopy"}
            String denoting the labeller type.
        **labeller_options
            Keyword arguments that are passed to the constructor for the given
            labeller type.
        """
        if labeller_type == "tomotopy":
            self.labeller = ignis.labeller.tomotopy.TomotopyLabeller(
                self.ignis_model.model, **labeller_options
            )
        else:
            raise ValueError(f"Unknown labeller type: '{labeller_type}'")

    def get_topic_labels(self, topic_id, top_n):
        """
        See `ignis.labeller.tomotopy.TomotopyLabeller.get_topic_labels()`.
        """
        if self.labeller is None:
            raise RuntimeError(
                "There is no labeller instantiated for this Aurum object. "
                "Use `.init_labeller()` to prepare one."
            )
        return self.labeller.get_topic_labels(topic_id, top_n)

    # =================================================================================
    # Visualisation Data
    # TODO: Move `vis_data` into a full visualisation object, like the labeller/model?
    def init_vis(self, vis_type, force=False, **vis_options):
        """
        Prepares the visualisation data for this `Aurum` object in the specified
        format.

        Parameters
        ----------
        vis_type: {"pyldavis", "clear"}
            String denoting the visualisation type.  Passing `"clear"` will remove
            any existing visualisation data.
        force: bool, optional
            Forces recalculation of `self.vis_data`, if it already exists.
        **vis_options
            Keyword arguments that are passed to the constructor for the given
            visualisation type.
        """
        if vis_type == "clear":
            self.vis_type = None
            self.vis_options = None
            self.vis_data = None
        elif vis_type == "pyldavis":
            if self.vis_data is not None and not force:
                raise RuntimeError(
                    f"Visualisation data already exists for this Aurum object "
                    f"(type: '{self.vis_type}'). "
                    f"Pass `force=True` to force recalculation."
                )

            self.vis_type = vis_type
            self.vis_options = vis_options
            self.vis_data = ignis.vis.pyldavis.prepare_data(
                self.ignis_model.model, **vis_options
            )
        else:
            raise ValueError(f"Unknown visualisation type: '{vis_type}'")

    def get_vis_data(self):
        """
        Returns the prepared visualisation data for this model, if any.

        Different visualisation classes may have different ways of storing and
        representing this data.
        """
        if self.vis_data is None:
            raise RuntimeError(
                "There is no visualisation data instantiated for this Aurum object. "
                "Use `.init_vis()` to prepare it."
            )
        return self.vis_data

    def show_visualisation(self, **kwargs):
        """
        Displays the prepared visualisation for this model.

        Presumes that the `ignis.vis` class returns a situation-appropriate format
        (e.g., for display in a Jupyter notebook)

        _Sample (using `pyLDAvis`):_

        .. image:: /ignis-tm/images/show_visualisation_pyldavis.png
           `show_visualisation()` widget screenshot (pyLDAvis)

        Parameters
        ----------
        **kwargs
            Passed through to the visualisation module

        Returns
        -------
        The output of the `show_visualisation` method of the relevant `ignis.vis` class.
        """
        vis_data = self.get_vis_data()

        if self.vis_type == "pyldavis":
            return ignis.vis.pyldavis.show_visualisation(vis_data, **kwargs)
        else:
            raise RuntimeError(f"Unknown saved visualisation type: '{self.vis_type}'")

    def get_visualisation_html(self, **kwargs):
        """
        Gets the prepared visualisation for this model as a raw HTML string.

        Parameters
        ----------
        **kwargs
            Passed through to the visualisation module.

        Returns
        -------
        The output of the `get_visualisation_html` method of the relevant `ignis.vis`
        class.
        """
        vis_data = self.get_vis_data()

        if self.vis_type == "pyldavis":
            return ignis.vis.pyldavis.get_visualisation_html(vis_data, **kwargs)
        else:
            raise RuntimeError(f"Unknown saved visualisation type: '{self.vis_type}'")

    def export_visualisation(self, folder, use_cdn=True):
        """
        Exports the visualisation prepared for this `Aurum` object as a standalone
        webpage to the given folder.

        If `use_cdn` is `True` (the default case), only a single HTML file will be
        generated, and additional JS/CSS sources will be loaded from a CDN instead.

        Otherwise, the additional JS/CSS sources will be exported directly into the
        target folder together with the main visualisation file -- Be sure to include
        the entire exported folder when moving it to a different location for display
        (e.g., on a PC with no internet access).

        Parameters
        ----------
        folder: str or pathlib.Path
            The folder to export the visualisation to.
        use_cdn: bool, optional
            If True, will save a single HTML file and attempt to load additional JS/CSS
            sources from a CDN instead.
        """
        vis_data = self.get_vis_data()

        # Assuming the `vis_data` get is successful, we should have a valid record of
        # the `vis_type` as well.
        if self.vis_type == "pyldavis":
            ignis.vis.pyldavis.export_visualisation(vis_data, folder, use_cdn)
        else:
            raise RuntimeError(f"Unknown saved visualisation type: '{self.vis_type}'")

    # =================================================================================
    # Jupyter widgets
    def nb_explore_topics(
        self,
        top_words=30,
        top_labels=15,
        doc_sort_key=None,
        display_fn=None,
        max_display_length=50000,
        metadata_full_doc_link="filename",
    ):
        """
        Convenience function that creates an interactive Jupyter notebook widget for
        exploring the topics and `ignis.corpus.Document` objects tracked by this model.

        By default, `ignis.corpus.Document` objects are displayed in decreasing order
        of probability for each specified topic, but a custom sorting function can be
        passed via `doc_sort_key` as well.

        Suggested topic labels will be shown if the model has a labeller initialised.

        _Sample:_

        .. image:: /ignis-tm/images/nb_explore_topics.png
           `nb_explore_topics()` widget screenshot

        Parameters
        ----------
        top_words: int, optional
            The top `n` most probable terms for each topic to show.
        top_labels: int, optional
            The top `n` most probable labels for each topic to show.

            Will have no effect if the model does not have a labeller initialised.
        doc_sort_key: fn, optional
            If specified, will sort `ignis.corpus.Document` objects for each topic
            using this key when displaying them.
        display_fn: fn, optional
            Custom display function that receives an individual
            `ignis.corpus.Document` as input, and should display the
            `ignis.corpus.Document` in human-readable form as a side effect.

            If unset, will assume that the human-readable representation of the
            `ignis.corpus.Document` is in HTML format and display it accordingly.
        max_display_length: int, optional
            Maximum length (in characters) to display from the
            `ignis.corpus.Document` object's human-readable representation.  If the
            `ignis.corpus.Document` object's human-readable representation is longer
            than this limit, a link will be generated to view the full human-readable
            representation in a new window.

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

        # Per topic info
        def show_topic(topic_id=1):
            # Top tokens
            words = ", ".join(
                word
                for word, probability in self.get_topic_words(topic_id, top_n=top_words)
            )
            print(f"Top tokens:\n{words}")

            # Labels
            if self.labeller is not None:
                labels = ", ".join(
                    label
                    for label, score in self.get_topic_labels(
                        topic_id, top_n=top_labels
                    )
                )
                print(f"\nSuggested labels:\n{labels}")

            top_n_header = ipywidgets.Output()
            # noinspection PyTypeChecker
            display(top_n_header)

            # Topic documents -- `within_top_n`
            def show_topic_doc(within_top_n=1):
                with top_n_header:
                    top_n_header.clear_output(wait=True)
                    # noinspection PyTypeChecker
                    display(
                        HTML(
                            f"<h4>Documents with Topic {topic_id} in the top "
                            f"{within_top_n} topic(s)</h4>"
                        )
                    )

                # Grab the documents that match the params passed, sorted by topic
                # probability in descending order
                topic_probs = self.get_topic_documents(topic_id, within_top_n)
                topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)

                topic_docs = [doc_id for doc_id, prob in topic_probs]

                if len(topic_docs) == 0:
                    print(
                        "No documents matching the given topic parameters. "
                        "Try increasing `n`, or reducing the number of total topics "
                        "and retraining the model."
                    )
                    return

                topic_docs = [self.get_document(doc_id) for doc_id in topic_docs]

                if doc_sort_key is not None:
                    topic_docs = sorted(topic_docs, key=doc_sort_key)

                # Show actual document
                def show_doc(index=1):
                    # Start `index` from 1 for user-friendliness
                    print(f"[Total documents: {len(topic_docs)}]\n")
                    doc = topic_docs[index - 1]

                    if display_fn is None:
                        # Default HTML display
                        print(f"ID: {doc.id}")
                        if "filename" in doc.metadata:
                            print(f"Filename: {doc.metadata['filename']}")
                        if "txt_filename" in doc.metadata:
                            print(f"Processed: {doc.metadata['txt_filename']}")

                        if "sender" in doc.metadata:
                            print(f"Sender: {doc.metadata['sender']}")
                        if "recipients" in doc.metadata:
                            recipients = doc.metadata["recipients"]

                            # Truncate long recipient lists for display
                            # (TODO: Make this optional?)
                            if len(recipients) > 5:
                                recipients = recipients[:5] + ["..."]

                            print(f"Recipients:\n{json.dumps(recipients, indent=2)}")

                        # Jupyter notebooks will interpret anything between $ signs
                        # as LaTeX formulae when rendering HTML output, so we need to
                        # replace them with escaped $ signs (only in Jupyter
                        # environments)
                        display_str = doc.display_str.replace("$", r"\$")

                        # Length check
                        if len(display_str) > max_display_length:
                            display_str = (
                                f"<p>"
                                f"<b>Document too long to display in full - "
                                f"Showing first {max_display_length} "
                                f"characters.</b>"
                                f"</p>"
                                f"<p>"
                                f"<a href="
                                f"'{doc.metadata[metadata_full_doc_link]}' "
                                f"target='_blank'>"
                                f"Click here"
                                f"</a> to open the full document in a new "
                                f"tab/window."
                                f"</p>"
                                f"<hr/>" + display_str[:max_display_length]
                            )

                        # noinspection PyTypeChecker
                        display(HTML(display_str))
                    else:
                        # User-provided display function
                        display_fn(doc)
                    print()
                    print(
                        "Top 10 document topics (in descending order of probability):"
                    )
                    top_topics = self.get_document_topics(doc.id, 10)
                    # Use a more readable format
                    for topic, prob in top_topics:
                        print(f"Topic {topic}: {prob*100:.2f}%")

                # Control and output widgets for the document viewer
                slider = ipywidgets.IntSlider(
                    description="Document",
                    min=1,
                    max=len(topic_docs),
                    continuous_update=False,
                    layout=ignis.util.jupyter_styles.slider_layout,
                    style=ignis.util.jupyter_styles.slider_style,
                )
                text = ipywidgets.BoundedIntText(
                    min=1,
                    max=len(topic_docs),
                    layout=ignis.util.jupyter_styles.slider_text_layout,
                )
                ipywidgets.jslink((slider, "value"), (text, "value"))
                ui = ipywidgets.HBox([slider, text])
                out = ipywidgets.interactive_output(show_doc, {"index": slider})
                # noinspection PyTypeChecker
                display(ui, out)

            # Control and output widgets for the document top-n-topics viewer
            slider = ipywidgets.IntSlider(
                description="No. of topics to consider per doc",
                min=1,
                max=self.get_num_topics(),
                continuous_update=False,
                layout=ignis.util.jupyter_styles.slider_layout,
                style=ignis.util.jupyter_styles.slider_style,
            )
            text = ipywidgets.BoundedIntText(
                min=1,
                max=self.get_num_topics(),
                layout=ignis.util.jupyter_styles.slider_text_layout,
            )
            ipywidgets.jslink((slider, "value"), (text, "value"))
            ui = ipywidgets.HBox([slider, text])
            out = ipywidgets.interactive_output(
                show_topic_doc, {"within_top_n": slider}
            )
            # noinspection PyTypeChecker
            display(ui, out)

        # Control and output widgets for the topic viewer
        slider = ipywidgets.IntSlider(
            description="Topic",
            min=1,
            max=self.get_num_topics(),
            continuous_update=False,
            layout=ignis.util.jupyter_styles.slider_layout,
            style=ignis.util.jupyter_styles.slider_style,
        )
        text = ipywidgets.BoundedIntText(
            min=1,
            max=self.get_num_topics(),
            layout=ignis.util.jupyter_styles.slider_text_layout,
        )
        ipywidgets.jslink((slider, "value"), (text, "value"))
        ui = ipywidgets.HBox([slider, text])
        out = ipywidgets.interactive_output(show_topic, {"topic_id": slider})
        # noinspection PyTypeChecker
        display(ui, out)

    # =================================================================================
    # Slicing and Iteration
    # Convenience functions that help with the exploring the Model-Corpus interface
    def slice_by_topics(self, topic_ids, within_top_n=1, ignore_topics=None):
        """
        General form of `Aurum.slice_by_topic()`.

        Parameters
        ----------
        topic_ids: iterable of int
            The IDs of the topics to consider.
        within_top_n: int, optional
            How many of each `ignis.corpus.Document` object's top topics to consider.
        ignore_topics: iterable of int, optional
            Don't count any of these topics if they are within the top `n` for a
            `ignis.corpus.Document`.

            E.g., for a `ignis.corpus.Document` with top topics `[5, 1, 3, ...]`,
            setting `ignore_topics` to `[5]` will consider the document's top topics
            to be `[1, 3, ...]` instead.

        Returns
        -------
        ignis.corpus.CorpusSlice
        """
        if ignore_topics is None:
            ignore_topics = []

        all_doc_ids = []
        for doc_id in self.document_ids:
            # This is a list of (<topic>, <prob>)
            doc_topics = self.get_document_topics(
                doc_id, within_top_n + len(ignore_topics)
            )
            doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
            checked_topics = 0
            for topic, prob in doc_topics:
                if topic not in ignore_topics:
                    # We've seen one more topic for this document
                    checked_topics += 1

                if checked_topics > within_top_n:
                    # Exceeded the topic check limit for this document
                    break

                if topic in topic_ids:
                    # Add it and go to the next document, we're done here
                    all_doc_ids.append(doc_id)
                    break

        return self.slice_by_ids(all_doc_ids)

    def slice_by_topic(self, topic_id, within_top_n=1, ignore_topics=None):
        """
        Convenience function to create a new `ignis.corpus.CorpusSlice` containing the
        `ignis.corpus.Document` objects that have the given topic as one of their top
        `n` topics under the current model.

        **NOTE**: `topic_id` is 1-indexed, not 0-indexed (i.e., it is in `range(1,
        len(topics) + 1)`).

        Parameters
        ----------
        topic_id: int
            The ID of the topic to consider.
        within_top_n: int, optional
            How many of each `ignis.corpus.Document` object's top topics to consider.
        ignore_topics: iterable of int, optional
            Don't count any of these topics if they are within the top `n` for a
            `ignis.corpus.Document`.

            E.g., for a `ignis.corpus.Document` with top topics `[5, 1, 3, ...]`,
            setting `ignore_topics` to `[5]` will consider the document's top topics
            to be `[1, 3, ...]` instead.

        Returns
        -------
        ignis.corpus.CorpusSlice
        """
        return self.slice_by_topics([topic_id], within_top_n, ignore_topics)

    def retrain_model(
        self,
        corpus_slice=None,
        model_type=None,
        model_options=None,
        labeller_type=None,
        labeller_options=None,
        vis_type=None,
        vis_options=None,
    ):
        """
        (Re-)trains a topic model over some `ignis.corpus.CorpusSlice`.

        If `model_type`, `labeller_type`, and/or `vis_type` are `None`, the saved
        options from this current `Aurum` object are carried over and used.

        If new `model_options`, `labeller_options`, and/or `vis_options` dictionaries
        are passed, they are _merged_ with the ones from this current `Aurum` object
        instead of replacing them outright.

        See `ignis.probat.train_model()` for details on the various parameters.

        Parameters
        ----------
        corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice, optional
            The `ignis.corpus.CorpusSlice` to (re-)train the model for.
        model_type: str, optional
            The type of topic model to train.
        model_options: dict, optional
            If set, will be merged with this `Aurum` object's saved options.
        labeller_type: str, optional
            The type of automated labeller to train.
        labeller_options: dict, optional
            If set, will be merged with this `Aurum` object's saved options.
        vis_type: str, optional
            The type of visualisation to prepare.
        vis_options: dict, optional
            If set, will be merged with this `Aurum` object's saved options.

        Returns
        -------
        ignis.aurum.Aurum
            The results object for the newly-trained model, which can be used for
            further exploration and iteration.
        """
        if corpus_slice is not None and len(corpus_slice) == 0:
            raise RuntimeError("Cannot retrain model on an empty CorpusSlice.")

        new_kwargs = {
            "corpus_slice": corpus_slice or self.corpus_slice,
            "model_type": model_type or self.model_type,
            "vis_type": vis_type or self.vis_type,
        }

        # We can only look up current labeller settings if this object has a labeller
        # initialised in the first place
        if self.labeller is not None:
            if labeller_type is None:
                new_kwargs["labeller_type"] = self.labeller.labeller_type
            if labeller_options is None:
                new_kwargs["labeller_options"] = self.labeller.options

        # Merge option dictionaries, where available
        if model_options is not None:
            new_kwargs["model_options"] = dict(self.model_options, **model_options)
        else:
            new_kwargs["model_options"] = self.model_options

        if labeller_options is not None:
            if self.labeller is not None:
                new_kwargs["labeller_options"] = dict(
                    self.labeller.options, **labeller_options
                )
            else:
                new_kwargs["labeller_options"] = labeller_options

        if vis_options is not None:
            if self.vis_options is not None:
                new_kwargs["vis_options"] = dict(self.vis_options, **vis_options)
            else:
                new_kwargs["vis_options"] = vis_options
        else:
            new_kwargs["vis_options"] = self.vis_options

        return ignis.probat.train_model(**new_kwargs)

    def resuggest_num_topics(
        self, corpus_slice=None, model_options=None, *args, **kwargs
    ):
        """
        (Re-)suggests a possible number of topics for some
        `ignis.corpus.CorpusSlice`, using this `Aurum` object's saved options as
        defaults.

        If a new model options dict is passed, it will be _merged_ with the one from
        this current `Aurum` object instead of replacing it outright.

        All other parameters, including coherence calculation options, are passed
        through directly to `ignis.probat.suggest_num_topics()`.

        Parameters
        ----------
        corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice, optional
            The new slice to suggest a number of topics for.  If `None`, will use
            this `Aurum` object's current `ignis.corpus.CorpusSlice`.
        model_options: dict, optional
            Any options set here will be merged with the current model options.
        *args, **kwargs
            Passed on to `ignis.probat.suggest_num_topics()`.

        Returns
        -------
        int
            Suggested topic count.
        """
        if corpus_slice is not None and len(corpus_slice) == 0:
            raise RuntimeError("Cannot retrain model on an empty CorpusSlice.")

        # The only options that are inherited directly from this `Aurum` instance are
        # `corpus_slice` and `model_options` (where appropriate)
        new_kwargs = {"corpus_slice": corpus_slice or self.corpus_slice}

        # Merge option dictionaries, where available
        if model_options is not None:
            new_kwargs["model_options"] = dict(self.model_options, **model_options)
        else:
            new_kwargs["model_options"] = self.model_options

        # All other arguments can be passed straight to
        # `ignis.probat.suggest_num_topics()`, if set
        new_kwargs = dict(kwargs, **new_kwargs)

        return ignis.probat.suggest_num_topics(*args, **new_kwargs)


def load_results(filename):
    """
    Loads an `ignis.aurum.Aurum` results object from the given file.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the `ignis.aurum.Aurum` object from.

    Returns
    -------
    ignis.aurum.Aurum
    """
    with bz2.open(filename, "rb") as fp:
        save_object = pickle.load(fp)

    # Rehydrate the Ignis/external models
    model_type = save_object["model_type"]
    save_model = save_object["save_model"]
    external_model_bytes = save_object["external_model_bytes"]

    if model_type[:3] == "tp_":
        # Tomotopy model
        external_model = _load_tomotopy_model(model_type, external_model_bytes)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    save_model.model = external_model

    # Rehydrate the Aurum object
    aurum = Aurum(save_model)

    aurum.vis_type = save_object["vis_type"]
    aurum.vis_options = save_object["vis_options"]
    aurum.vis_data = save_object["vis_data"]

    return aurum


def _load_tomotopy_model(model_type, model_bytes):
    """
    Loads a Tomotopy model of the specified type from its binary representation.

    (All Tomotopy models are subclasses of tomotopy.LDAModel)

    Parameters
    ----------
    model_type: {"tp_lda"}
        String identifying the type of the saved Tomotopy model
    model_bytes: bytes
        The actual saved model

    Returns
    -------
    tp.LDAModel
    """
    import ignis.models

    if model_type == "tp_lda":
        return ignis.models.LDAModel.load_from_bytes(model_bytes)
    elif model_type == "tp_hdp":
        return ignis.models.HDPModel.load_from_bytes(model_bytes)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


def show_visualisations(aurum_objects):
    """
    Given some list of multiple `ignis.aurum.Aurum` objects that have initialised
    visualisation data, show them sequentially with a Jupyter notebook slider widget.

    _Sample (using `pyLDAvis` visualisations):_

    .. image:: /ignis-tm/images/show_visualisations_pyldavis.png
       `show_visualisations()` widget screenshot (pyLDAvis)

    Parameters
    ----------
    aurum_objects: iterable of `ignis.aurum.Aurum`
        The result `ignis.aurum.Aurum` objects to display visualisations for.
    """
    import ipywidgets
    from IPython.core.display import display, HTML
    import ignis.util.jupyter_styles

    # Control widgets
    slider = ipywidgets.IntSlider(
        description="Visualisation:",
        min=1,
        max=len(aurum_objects),
        continuous_update=False,
        layout=ignis.util.jupyter_styles.slider_layout,
        style=ignis.util.jupyter_styles.slider_style,
    )
    text = ipywidgets.BoundedIntText(
        min=1,
        max=len(aurum_objects),
        layout=ignis.util.jupyter_styles.slider_text_layout,
    )
    ipywidgets.jslink((slider, "value"), (text, "value"))

    # Display function
    def show_vis(index=1):
        # `index` starts at 1 for user-friendliness
        aurum_object = aurum_objects[index - 1]

        header_html = f"<h2 style='margin-bottom: 1em;'>Visualisation {index}</h2>"
        vis_html = aurum_object.get_visualisation_html()

        # noinspection PyTypeChecker
        display(HTML(header_html + vis_html))

    # Show widgets
    ui = ipywidgets.HBox([slider, text])
    out = ipywidgets.interactive_output(show_vis, {"index": slider})
    # noinspection PyTypeChecker
    display(ui, out)

    # Show the initial output (i.e., with index == 1).  We need to call it manually
    # here because some visualisations (e.g., pyLDAvis) will ony be drawn if `out`
    # has already been made visible via `display()`.
    with out:
        out.clear_output(wait=True)
        show_vis()
