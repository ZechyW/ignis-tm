import bz2
import copy
import pathlib
import pickle
import tempfile
import tomotopy as tp


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

    def __init__(self, corpus_slice, model):
        self.corpus_slice = corpus_slice
        self.model = model

    def save(self, filename):
        """
        Saves the Aurum object, including its associated Ignis model, to the given file.

        Essentially uses a compressed Pickle format.

        Parameters
        ----------
        filename: str or pathlib.Path
            The file to save the model to
        """
        filename = pathlib.Path(filename)

        # Copy the Ignis model, separate the actual Tomotopy part out, pickle
        # everything together
        external_model = self.model.model
        self.model.model = None
        save_model = copy.deepcopy(self.model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = pathlib.Path(tmpdir) / "save_model.bin"
            # model.save() expects the filename to be a string
            tmp_model_file = str(tmp_model_file)
            external_model.save(tmp_model_file)
            with open(tmp_model_file, "rb") as fp:
                external_model_bytes = fp.read()

        save_object = {
            "corpus_slice": self.corpus_slice,
            "model_type": self.model.model_type,
            "save_model": save_model,
            "external_model_bytes": external_model_bytes,
        }

        with bz2.open(filename, "wb") as fp:
            pickle.dump(save_object, fp)

        self.model.model = external_model


def load(filename):
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

    external_model = None
    if model_type == "tp_lda":
        # Tomotopy LDA model
        external_model = _load_tomotopy_model(tp.LDAModel, external_model_bytes)

    save_model.model = external_model

    return Aurum(corpus_slice, save_model)


def _load_tomotopy_model(tp_class, model_bytes):
    """
    Loads a Tomotopy model of the specified type from its binary representation.

    (All Tomotopy models are subclasses of tomotopy.LDAModel)

    Parameters
    ----------
    tp_class: type
    model_bytes: bytes

    Returns
    -------
    tp.LDAModel
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_file = pathlib.Path(tmpdir) / "load_model.bin"
        # model.save() expects the filename to be a string
        tmp_model_file = str(tmp_model_file)
        with open(tmp_model_file, "wb") as fp:
            fp.write(model_bytes)

        tp_model = tp_class.load(tmp_model_file)

    return tp_model
