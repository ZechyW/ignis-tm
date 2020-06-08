import bz2
import copy
import pathlib
import pickle
import pprint
import tempfile

import tomotopy as tp

import ignis.labeller.tomotopy
import ignis.vis.pyldavis


class Aurum:
    """
    Aurum instances hold the results of performing topic modelling over Documents.
    They provide methods for easily exploring the results and iterating over the topic
    modelling process.

    Aurum objects basically bring together the public APIs for Ignis models,
    automated labellers, and visualisation data providers, while also providing general
    save/load functionality.

    NOTE: All topic IDs retrieved from Aurum instances are 1-indexed rather than
    0-indexed. So a model with 5 topics has topic IDs [1, 2, 3, 4, 5] and not
    [0, 1, 2, 3, 4].

    This is for easier matching against pyLDAvis visualisations, and for easier usage
    by non-technical users.

    Parameters
    ----------
    ignis_model: ignis.models.BaseModel
        The specific Ignis topic model that was trained
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

    def get_topic_documents(self, topic_id, within_top_n):
        """
        See `ignis.models.base.BaseModel.get_topic_documents()`
        """
        return self.ignis_model.get_topic_documents(topic_id, within_top_n)

    def get_document_topics(self, doc_id, top_n):
        """
        See `ignis.models.base.BaseModel.get_document_topics()`
        """
        return self.ignis_model.get_document_topics(doc_id, top_n)

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

    def get_document(self, doc_id):
        """
        See `ignis.corpus.CorpusSlice.get_document()`
        """
        return self.corpus_slice.get_document(doc_id)

    def slice_by_ids(self, doc_ids):
        """
        See `ignis.corpus.CorpusSlice.slice_by_ids()`
        """
        return self.corpus_slice.slice_by_ids(doc_ids)

    def slice_by_tokens(self, tokens, include_root):
        """
        See `ignis.corpus.CorpusSlice.slice_by_tokens()`
        """
        return self.corpus_slice.slice_by_tokens(tokens, include_root)

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
        vis_type: {"pyldavis"}
            String denoting the visualisation type.
        force: bool, optional
            If `self.vis_data` is already set, it will not be recalculated unless
            `force` is set.
        **vis_options
            Keyword arguments that are passed to the constructor for the given
            visualisation type.
        """
        if vis_type == "pyldavis":
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
        The output of the `show_visualisation` method of the relevant `ingnis.vis` class
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
    def nb_show_topics(self, top_labels=None, top_words=None):
        """
        Convenience function to create an interactive Jupyter notebook widget for
        exploring the topics in this model.

        Parameters
        ----------
        top_labels: int, optional
            Number of top suggested labels for this topic to show.
            If None, will skip showing labels.
        top_words: int, optional
            Number of top words for this topics to show.
            If None, will skip showing top words.

        Returns
        -------
        ipywidgets.interact function
        """
        import ipywidgets

        # Prepare the display function
        def show_topic(topic_id=1):
            print(f"[Topic {topic_id}]")

            # Labels
            if top_labels is not None:
                labels = ", ".join(
                    label
                    for label, score in self.get_topic_labels(
                        topic_id, top_n=top_labels
                    )
                )
                print(f"\nSuggested labels:\n{labels}")

            # Top words
            if top_words is not None:
                words_probs = self.get_topic_words(topic_id, top_n=top_words)
                words = [x[0] for x in words_probs]

                words = ", ".join(words)
                print(f"\nTop words:\n{words}")

        return ipywidgets.interact(show_topic, topic_id=(1, self.get_num_topics()))

    def nb_show_topic_documents(self, topic_id, within_top_n):
        """
        Convenience function to create an interactive Jupyter notebook widget for
        exploring the documents under a certain topic in the current model.

        See `ignis.models.base.BaseModel.get_topic_documents()` for details on the
        parameters accepted.

        Note that `topic_id` starts from 1 and not 0.

        Parameters
        ----------
        topic_id
        within_top_n

        Returns
        -------
        ipywidgets.interact function
        """
        import ipywidgets

        # Grab the documents that match the params passed
        topic_docs = [
            doc_id
            for doc_id, prob in self.get_topic_documents(
                topic_id=topic_id, within_top_n=within_top_n
            )
        ]

        # Prepare the display function
        def show_topic_doc(index=0):
            doc_id = topic_docs[index]
            doc = self.get_document(doc_id)
            print(str(doc))
            print()
            print("Top document topics (in descending order of probability):")
            pprint.pprint(self.get_document_topics(doc_id, 10))

        return ipywidgets.interact(show_topic_doc, index=(0, len(topic_docs) - 1))

    # =================================================================================
    # Slicing and Iteration
    # Convenience functions that help with the exploring the Model-Corpus interface
    def slice_by_topic(self, topic_id, within_top_n):
        """
        Convenience function to create a new CorpusSlice with Documents that come
        under a given topic in the current model.

        See `ignis.models.base.BaseModel.get_topic_documents()` for details on the
        parameters accepted.

        Note that `topic_id` starts from 1 and not 0.

        Parameters
        ----------
        topic_id
        within_top_n

        Returns
        -------
        ignis.corpus.CorpusSlice
        """
        topic_docs = [
            doc_id
            for doc_id, prob in self.get_topic_documents(
                topic_id=topic_id, within_top_n=within_top_n
            )
        ]
        return self.slice_by_ids(topic_docs)

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
    if model_type == "tp_lda":
        tp_class = tp.LDAModel
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_file = pathlib.Path(tmpdir) / "load_model.bin"
        # model.save() expects the filename to be a string
        tmp_model_file = str(tmp_model_file)
        with open(tmp_model_file, "wb") as fp:
            fp.write(model_bytes)

        # noinspection PyTypeChecker,PyCallByClass
        tp_model = tp_class.load(tmp_model_file)

    return tp_model
