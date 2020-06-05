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

    Parameters
    ----------
    corpus_slice: ignis.corpus.CorpusSlice
        The CorpusSlice that was topic-modelled
    model: ignis.models.BaseModel
        The specific topic model that was trained
    """

    def __init__(self, corpus_slice, ignis_model):
        self.corpus_slice = corpus_slice
        self.ignis_model = ignis_model

        # Aurum objects also optionally have cached labeller and visualisation data
        # objects
        self.labeller = None
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

        # Copy the Ignis model, separate the actual Tomotopy part out, pickle
        # everything together
        external_model = self.ignis_model.model
        self.ignis_model.model = None
        save_model = copy.deepcopy(self.ignis_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "save_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            external_model.save(tmp_model_file)
            with open(tmp_model_file, "rb") as fp:
                external_model_bytes = fp.read()

        save_object = {
            "corpus_slice": self.corpus_slice,
            "save_model": save_model,
            "model_type": save_model.model_type,
            "external_model_bytes": external_model_bytes,
            # We should also be able to save any cached visualisation data, but the
            # labeller is probably not pickle-able.
            "vis_data": self.vis_data,
        }

        with bz2.open(filename, "wb") as fp:
            pickle.dump(save_object, fp)

        self.ignis_model.model = external_model

    def init_labeller(self, labeller_type, labeller_options):
        """
        Trains an automated labeller for this Aurum object

        Parameters
        ----------
        labeller_type: {"tomotopy"}
            String denoting the labeller type.
        labeller_options: dict, optional
            Dictionary of options for the given labeller type.
        """
        if labeller_type == "tomotopy":
            self.labeller = ignis.labeller.tomotopy.TomotopyLabeller(
                self.ignis_model.model, labeller_options
            )
        else:
            raise ValueError(f"Unknown labeller type: '{labeller_type}'")

    def get_topic_labels(self, *args, **kwargs):
        """
        Passes arguments directly through to the labeller.
        """
        if self.labeller is None:
            raise RuntimeError(
                "There is no labeller instantiated for this Aurum object. "
                "Use `.init_labeller()` to prepare one."
            )
        return self.labeller.get_topic_labels(*args, **kwargs)

    def init_vis(self, vis_type, vis_options):
        """
        Prepares a visualisation for this Aurum object in the given format

        Parameters
        ----------
        vis_type: {"pyldavis"}
            String denoting the visualisation type.
        vis_options: dict, optional
            Dictionary of options for the given visualisation type.
        """
        if vis_type == "pyldavis":
            self.vis_data = ignis.vis.pyldavis.prepare_data(
                self.ignis_model.model, vis_options
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


def load_model(filename):
    """
    Loads an Aurum object from the given file.

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

    corpus_slice = save_object["corpus_slice"]
    model_type = save_object["model_type"]
    save_model = save_object["save_model"]
    external_model_bytes = save_object["external_model_bytes"]

    vis_data = save_object["vis_data"]

    if model_type[:3] == "tp_":
        # Tomotopy model
        external_model = _load_tomotopy_model(model_type, external_model_bytes)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    save_model.model = external_model

    aurum = Aurum(corpus_slice, save_model)
    aurum.vis_data = vis_data

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
