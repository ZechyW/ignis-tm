"""
The methods in this class are used for performing the actual topic modelling to get
Aurum results.
"""
import ignis
import ignis.aurum
import ignis.corpus


def train_model(
    corpus_slice,
    model_type="lda",
    model_options=None,
    prepare_labeller=False,
    prepare_vis=False,
):
    """
    Top-level helper for training topic models using various algorithms

    Parameters
    ----------
    corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice
        The CorpusSlice to perform the topic modelling over.  If a Corpus is passed
        instead, a CorpusSlice containing all of its Documents will be created.
    model_type: {"lda", "hdp"}
        Type of model to train
    model_options: dict, optional
        Dictionary of keyword arguments that will be passed to the relevant
        `ignis.models` model constructor
    prepare_labeller: bool, optional
        Whether or not to train the automated labeller, if available
    prepare_vis: bool, optional
        Whether or not to prepare the visualisation data, if available

    Returns
    -------
    ignis.aurum.Aurum
        The Aurum results object for the trained model, which can be used for further
        exploration and iteration
    """
    if isinstance(corpus_slice, ignis.corpus.Corpus):
        corpus_slice = corpus_slice.slice_full()
    if not isinstance(corpus_slice, ignis.corpus.CorpusSlice):
        raise ValueError(
            "Ignis models must be instantiated with Corpus or CorpusSlice instances."
        )

    if model_type == "lda":
        model = ignis.LDAModel(corpus_slice, model_options)
        model.train()
        return ignis.aurum.Aurum(corpus_slice, model)
    elif model_type == "hdp":
        pass
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
