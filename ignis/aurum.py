import bz2
import copy
import pathlib
import pickle
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

        # Grab a reference to the CorpusSlice object so that we can use its methods
        # directly
        self.corpus_slice = ignis_model.corpus_slice

        # Aurum objects also optionally have cached labeller and visualisation data
        # objects
        self.labeller = None

        self.vis_type = None
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
            self.vis_data = ignis.vis.pyldavis.prepare_data(
                self.ignis_model.model, **vis_options
            )
        else:
            raise ValueError(f"Unknown visualisation type: '{vis_type}'")

    def get_vis_data(self):
        """
        Returns the prepared visualisation data for this model, if any
        """
        if self.vis_data is None:
            raise RuntimeError(
                "There is no visualisation data instantiated for this Aurum object. "
                "Use `.init_vis()` to prepare it."
            )
        return self.vis_data

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
