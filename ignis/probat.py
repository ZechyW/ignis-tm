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
    labeller_type=None,
    labeller_options=None,
    vis_type=None,
    vis_options=None,
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
        Dictionary of options that will be passed to the relevant `ignis.models`
        model constructor
    labeller_type: {"tomotopy"}, optional
        The type of automated labeller to use, if available
    labeller_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.labeller`
        object constructor
    vis_type: {"pyldavis"}, optional
        The type of visualisation data to extract, if available
    vis_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.vis`
        object constructor

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
        aurum = ignis.aurum.Aurum(model)
    elif model_type == "hdp":
        aurum = None
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    if labeller_type is not None:
        aurum.init_labeller(labeller_type, **labeller_options)

    if vis_type is not None:
        aurum.init_vis(vis_type, **vis_options)

    return aurum
