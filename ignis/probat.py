"""
The methods in this class are used for performing the actual topic modelling to get
Aurum results.
"""
import tqdm

import ignis
import ignis.aurum
import ignis.corpus
import ignis.models


def train_model(
    corpus_slice,
    model_type="tp_lda",
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
    model_type: {"tp_lda", "tp_hdp"}
        Type of model to train; corresponds to the model type listed in the relevant
        `ignis.models` class
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

    if model_type == "tp_lda":
        model = ignis.models.LDAModel(corpus_slice, model_options)
        model.train()
        aurum = ignis.aurum.Aurum(model)
    elif model_type == "tp_hdp":
        model = ignis.models.HDPModel(corpus_slice, model_options)
        model.train()
        aurum = ignis.aurum.Aurum(model)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    if labeller_type is not None:
        if labeller_options is None:
            labeller_options = {}

        aurum.init_labeller(labeller_type, **labeller_options)

    if vis_type is not None:
        if vis_options is None:
            vis_options = {}

        aurum.init_vis(vis_type, **vis_options)

    return aurum


def suggest_num_topics(
    corpus_slice,
    model_type="tp_lda",
    model_options=None,
    coherence="u_mass",
    top_n=10,
    start_k=2,
    end_k=10,
    iterations=100,
    verbose=False,
):
    """
    Lightly trains models with various topic counts and uses the resultant coherence
    scores to suggest an optimal number of topics to use for full training.

    Parameters
    ----------
    corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice
        The CorpusSlice to perform the topic modelling over.  If a Corpus is passed
        instead, a CorpusSlice containing all of its Documents will be created.
    model_type: {"tp_lda", "tp_hdp"}
        Type of model to train; corresponds to the model type listed in the relevant
        `ignis.models` class
    model_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.models`
        model constructor
    coherence: {"u_mass", "c_v", "c_uci", "c_npmi"}
        Coherence measure to calculate
    top_n: int
        Number of top words to extract from each topic when measuring coherence
    start_k: int, optional
        Minimum topic count to consider
    end_k: int, optional
        Maximum topic count to consider
    iterations: int, optional
        Number of iterations to train each candidate model for
    verbose: bool, optional
        Whether or not to show interim training progress

    Returns
    -------
    int
        Suggested topic count
    """
    if isinstance(corpus_slice, ignis.corpus.Corpus):
        corpus_slice = corpus_slice.slice_full()
    if not isinstance(corpus_slice, ignis.corpus.CorpusSlice):
        raise ValueError(
            "Ignis models must be instantiated with Corpus or CorpusSlice instances."
        )

    candidate_counts = range(start_k, end_k + 1)

    progress_bar = None
    if verbose:
        total_models = end_k - start_k + 1
        print(
            f"Training {total_models} mini-models to suggest a suitable number of "
            f"topics between {start_k} and {end_k}..."
            f"({len(corpus_slice)} documents, {iterations} iterations each)"
        )
        progress_bar = tqdm.tqdm(total=total_models * iterations, miniters=1)

    results = []
    for k in candidate_counts:
        this_options = dict(
            model_options,
            k=k,
            iterations=iterations,
            update_every=iterations,
            until_max_ll=False,
            until_max_coherence=False,
            verbose=False,
        )

        if model_type == "tp_lda":
            model = ignis.models.LDAModel(corpus_slice, this_options)
            model.train()
            results.append(
                (
                    k,
                    model.get_coherence(
                        coherence=coherence,
                        top_n=top_n,
                        processes=model.options["workers"],
                    ),
                )
            )

        if verbose:
            progress_bar.update(iterations)

    results = sorted(results, key=lambda x: x[1], reverse=True)
    best = results[0]

    if verbose:
        print(
            f"Suggested topic count: {best[0]}\t"
            f"Coherence: {best[1]} (always negative, closer to 0 is better)"
        )
        progress_bar.close()

    return best[0]
