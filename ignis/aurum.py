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
    Aurum instances hold the results of performing topic modelling over
    :class:`~ignis.corpus.Document` instances. They provide methods for easily
    exploring the results and iterating over the topic modelling process.

    Aurum objects basically bring together the public APIs for Ignis models,
    automated labellers, and visualisation data providers, while also providing general
    save/load functionality.

    NOTE: All topic IDs retrieved from Aurum instances are 1-indexed rather than
    0-indexed. So a model with 5 topics has topic IDs `[1, 2, 3, 4, 5]` and not
    `[0, 1, 2, 3, 4]`.

    This is for easier matching against pyLDAvis visualisations, and for easier usage
    by non-technical users.

    Parameters
    ----------
    ignis_model: ignis.models.base.BaseModel
        The specific Ignis topic model used to generate this Aurum object.
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
        Saves the Aurum object, including its associated Ignis model, to the given file.
        Essentially uses a bz2-compressed Pickle format.

        Also attempts to save any cached visualisation data, but the labeller is
        probably not pickle-able.

        Parameters
        ----------
        filename: str or pathlib.Path
            File to save the model to
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
        See `ignis.models.base.BaseModel.get_num_topics()`
        """
        return self.ignis_model.get_num_topics()

    def get_topic_words(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_topic_words()`
        """
        return self.ignis_model.get_topic_words(*args, **kwargs)

    def get_topic_documents(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_topic_documents()`
        """
        return self.ignis_model.get_topic_documents(*args, **kwargs)

    def get_document_topics(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_document_topics()`
        """
        return self.ignis_model.get_document_topics(*args, **kwargs)

    def get_document_top_topic(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_document_top_topic()`
        """
        return self.ignis_model.get_document_top_topic(*args, **kwargs)

    def get_coherence(self, *args, **kwargs):
        """
        See `ignis.models.base.BaseModel.get_coherence()`
        """
        return self.ignis_model.get_coherence(*args, **kwargs)

    # =================================================================================
    # Corpus Slice
    def get_documents(self):
        """
        Get the IDs of all the documents that are covered by this Ignis model.

        Returns
        -------
        iterable of str
        """
        return list(self.corpus_slice.documents.keys())

    def get_document(self, *args, **kwargs):
        """
        See `ignis.corpus.CorpusSlice.get_document()`
        """
        return self.corpus_slice.get_document(*args, **kwargs)

    def slice_by_ids(self, doc_ids):
        """
        See `ignis.corpus.CorpusSlice.slice_by_ids()`
        """
        return self.corpus_slice.slice_by_ids(doc_ids)

    def slice_by_tokens(self, tokens, include_root=False, human_readable=False):
        """
        See `ignis.corpus.CorpusSlice.slice_by_tokens()`
        """
        return self.corpus_slice.slice_by_tokens(tokens, include_root, human_readable)

    def slice_without_tokens(self, tokens, include_root=False, human_readable=False):
        """
        See `ignis.corpus.CorpusSlice.slice_without_tokens()`
        """
        return self.corpus_slice.slice_without_tokens(
            tokens, include_root, human_readable
        )

    def slice_filter(self, filter_fn, include_root=False):
        """
        See `ignis.corpus.CorpusSlice.slice_filter()`
        """
        return self.corpus_slice.slice_filter(filter_fn, include_root)

    # =================================================================================
    # Automated Labeller
    def init_labeller(self, labeller_type, **labeller_options):
        """
        Trains an automated labeller for this Aurum object

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
        See `ignis.labeller.tomotopy.get_topic_labels()`
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
        Prepares a visualisation for this Aurum object in the given format

        Parameters
        ----------
        vis_type: {"pyldavis", "clear"}
            String denoting the visualisation type.  Setting to "clear" will remove
            any existing visualisation data.
        force: bool, optional
            If `self.vis_data` is already set, it will not be recalculated unless
            `force` is set.
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
        """
        if self.vis_data is None:
            raise RuntimeError(
                "There is no visualisation data instantiated for this Aurum object. "
                "Use `.init_vis()` to prepare it."
            )
        return self.vis_data

    def show_visualisation(self):
        """
        Displays the prepared visualisation for this model.
        (Presumes that the `ignis.vis` class returns a situation-appropriate format,
        e.g., for display in a Jupyter notebook)

        Returns
        -------
        The output of the `show_visualisation` method of the relevant `ignis.vis` class
        """
        vis_data = self.get_vis_data()

        if self.vis_type == "pyldavis":
            return ignis.vis.pyldavis.show_visualisation(vis_data)
        else:
            raise RuntimeError(f"Unknown saved visualisation type: '{self.vis_type}'")

    def export_visualisation(self, folder):
        """
        Exports the visualisation prepared for this Aurum object to the given folder.

        Parameters
        ----------
        folder
        """
        vis_data = self.get_vis_data()

        # Assuming the `vis_data` get is successful, we should have a valid record of
        # the `vis_type` as well.
        if self.vis_type == "pyldavis":
            ignis.vis.pyldavis.export_visualisation(vis_data, folder)
        else:
            raise RuntimeError(f"Unknown saved visualisation type: '{self.vis_type}'")

    # =================================================================================
    # Jupyter widgets
    def nb_explore_topics(
        self, top_words=30, top_labels=15, doc_sort_key=None, display_fn=None
    ):
        """
        Convenience function that creates an interactive Jupyter notebook widget for
        exploring the topics in this model.

        By default, documents are shown in decreasing order of probability for each
        specified topic, but a custom sorting function can be passed via `doc_sort_key`
        as well.

        Parameters
        ----------
        top_words: int, optional
            The top `n` most probable terms for each topic to show
        top_labels: int, optional
            The top `n` most probably labels for each topic to show.  Will have no
            effect if the model does not have a labeller initialised.
        doc_sort_key: fn, optional
            If specified, will sort topic documents using this key when displaying them.
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

        # Per topic info
        def show_topic(topic_id=1):
            # Styles
            # - Limit the height of most output areas for readability
            # - Prevent vertical scrollbars in nested output subareas
            jupyter_styles = """
            <style>
                div.cell > div.output_wrapper > div.output.output_scroll {
                    height: auto;
                }
            
                .jupyter-widgets-output-area .output_scroll {
                    height: unset;
                    border-radius: unset;
                    -webkit-box-shadow: unset;
                    box-shadow: unset;
                }
            
                .jupyter-widgets-output-area, .output_stdout, .output_result {
                    height: auto;
                    max-height: 50em;
                    overflow-y: auto;
                }
                .jupyter-widgets-output-area .jupyter-widgets-output-area {
                    max-height: unset;
                }
            </style>
            """
            # noinspection PyTypeChecker
            display(HTML(jupyter_styles))

            # Top words
            words = ", ".join(
                word
                for word, probability in self.get_topic_words(topic_id, top_n=top_words)
            )
            print(f"Top words:\n{words}")

            # Labels
            if self.labeller is not None:
                labels = ", ".join(
                    label
                    for label, score in self.get_topic_labels(
                        topic_id, top_n=top_labels
                    )
                )
                print(f"\nSuggested labels:\n{labels}")

            # Topic documents -- `within_top_n`
            # noinspection PyTypeChecker
            display(
                HTML(
                    f"<h4>Documents with Topic {topic_id} in the top <em>n</em> "
                    f"topics</h4>"
                )
            )

            def show_topic_doc(within_top_n=1):
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
                def show_doc(index=0):
                    print(f"[Total documents: {len(topic_docs)}]\n")
                    doc = topic_docs[index]

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

                        # noinspection PyTypeChecker
                        display(HTML(display_str))
                    else:
                        # User-provided display function
                        display_fn(doc)
                    print()
                    print("Top document topics (in descending order of probability):")
                    pprint.pprint(self.get_document_topics(doc.id, 10))

                return ipywidgets.interact(
                    show_doc,
                    index=ipywidgets.IntSlider(
                        description="Document", min=0, max=len(topic_docs) - 1
                    ),
                )

            return ipywidgets.interact(
                show_topic_doc,
                within_top_n=ipywidgets.IntSlider(
                    description="n", min=1, max=self.get_num_topics()
                ),
            )

        return ipywidgets.interact(
            show_topic,
            topic_id=ipywidgets.IntSlider(
                description="Topic", min=1, max=self.get_num_topics()
            ),
        )

    # =================================================================================
    # Slicing and Iteration
    # Convenience functions that help with the exploring the Model-Corpus interface
    def slice_by_topics(self, topic_ids, within_top_n=1, ignore_topics=None):
        """
        Convenience function to create a new CorpusSlice with Documents that come
        under all the given topics in the current model.

        See `ignis.models.base.BaseModel.get_topic_documents()` for details on the
        `within_top_n` parameter.

        Note that `topic_id` starts from 1 and not 0.

        Parameters
        ----------
        topic_ids: iterable of int
        within_top_n: int, optional
        ignore_topics: iterable of int, optional
            Don't count any of these topics if they are within the top `n` for a
            document.  E.g., for a document with top topics [5, 1, 3, ...], setting
            `ignore_topics` to [5] will consider the document's top topics to be [1,
            3, ...] instead.

        Returns
        -------
        ignis.corpus.CorpusSlice
        """
        if ignore_topics is None:
            ignore_topics = []

        all_doc_ids = []
        for doc_id in self.get_documents():
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
        Convenience function to create a new CorpusSlice with Documents that come
        under a given topic in the current model.

        See `ignis.models.base.BaseModel.get_topic_documents()` for details on the
        `within_top_n` and `ignore_topics` parameters.

        Note that `topic_id` starts from 1 and not 0.

        Parameters
        ----------
        topic_id: int
        within_top_n: int, optional
        ignore_topics: iterable of int, optional

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
        (Re-)trains a topic model with any number of specified options changed; any
        parameters that are None will be kept the same as the current model.

        See `ignis.probat.train_model()` for details on the params.

        Parameters
        ----------
        corpus_slice
        model_type
        model_options
        labeller_type
        labeller_options
        vis_type
        vis_options

        Returns
        -------
        ignis.aurum.Aurum
            The Aurum results object for the newly-trained model, which can be used
            for further exploration and iteration
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
        self,
        corpus_slice=None,
        model_type=None,
        model_options=None,
        coherence=None,
        top_n=None,
        start_k=None,
        end_k=None,
        iterations=None,
        verbose=False,
    ):
        """
        (Re-)suggests a possible number of topics for the given corpus slice and
        model type; any parameters that are None will be kept the same as the
        current model or set to the Ignis defaults, where appropriate.

        See `ignis.probat.suggest_num_topics()` for details on the params.

        Parameters
        ----------
        corpus_slice
        model_type
        model_options
        coherence
        top_n
        start_k
        end_k
        iterations
        verbose

        Returns
        -------
        int
            Suggested topic count
        """
        if corpus_slice is not None and len(corpus_slice) == 0:
            raise RuntimeError("Cannot retrain model on an empty CorpusSlice.")

        # The only options that are inherited from this Aurum instance are
        # `corpus_slice`, `model_type`, and `model_options`, where appropriate
        new_kwargs = {
            "corpus_slice": corpus_slice or self.corpus_slice,
            "model_type": model_type or self.model_type,
        }

        # Merge option dictionaries, where available
        if model_options is not None:
            new_kwargs["model_options"] = dict(self.model_options, **model_options)
        else:
            new_kwargs["model_options"] = self.model_options

        # All other arguments can be passed straight to
        # `ignis.probat.suggest_num_topics()`, if set
        if coherence is not None:
            new_kwargs["coherence"] = coherence
        if top_n is not None:
            new_kwargs["top_n"] = top_n
        if start_k is not None:
            new_kwargs["start_k"] = start_k
        if end_k is not None:
            new_kwargs["end_k"] = end_k
        if iterations is not None:
            new_kwargs["iterations"] = iterations

        new_kwargs["verbose"] = verbose

        return ignis.probat.suggest_num_topics(**new_kwargs)


def load_results(filename):
    """
    Loads an Aurum results object from the given file.

    Parameters
    ----------
    filename: str or pathlib.Path
        The file to load the Aurum object from.

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
