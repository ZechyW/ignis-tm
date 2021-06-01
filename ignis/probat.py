"""
Functions for performing topic modelling to get `ignis.aurum.Aurum` results.
"""
import time

from tqdm.auto import tqdm

import ignis
import ignis.aurum
import ignis.corpus
import ignis.models


def init_lda_model_wp(corpus_slice, model_options=None):
    """
    Prepare an `ignis.models.lda.LDAModel` for use with word priors.
    (contrib.: C. Ow)

    Parameters
    ----------
    corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice
        The `ignis.corpus.CorpusSlice` to train the model on.
    model_options: dict, optional
        Model-specific options.  See `ignis.models.lda.LDAModel` for details.

    Returns
    -------
    ignis.models.lda.LDAModel
    """
    if type(corpus_slice) is ignis.corpus.Corpus:
        corpus_slice = corpus_slice.slice_full()
    if not type(corpus_slice) is ignis.corpus.CorpusSlice:
        raise ValueError(
            "Ignis models must be instantiated with Corpus or CorpusSlice instances."
        )

    return ignis.models.LDAModel(corpus_slice, model_options)


def train_model(
    corpus_slice,
    pre_model=None,
    model_type="tp_lda",
    model_options=None,
    labeller_type=None,
    labeller_options=None,
    vis_type="pyldavis",
    vis_options=None,
):
    """
    Top-level helper for training topic models using the various algorithms available.

    Parameters
    ----------
    corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice
        The `ignis.corpus.CorpusSlice` to perform the topic modelling over.  If a
        `ignis.corpus.Corpus` is passed instead, a `ignis.corpus.CorpusSlice`
        containing all of its `ignis.corpus.Document` objects will be created.
    pre_model: ignis.models.lda.LDAModel, optional
        This is needed when you want to train a `tomotopy` LDA model with word priors.
        Default is `None`.
    model_type: {"tp_lda", "tp_hdp", "tp_lda_wp"}
        Type of model to train; corresponds to the model type listed in the relevant
        `ignis.models` class.
    model_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.models`
        model constructor.
    labeller_type: {"tomotopy"}, optional
        The type of automated labeller to use, if available.
    labeller_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.labeller`
        object constructor.
    vis_type: {"pyldavis"}, optional
        The type of visualisation data to extract, if available.
    vis_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.vis`
        object constructor.

    Returns
    -------
    ignis.aurum.Aurum
        The `ignis.aurum.Aurum` results object for the trained model, which can be used
        for further exploration and iteration.
    """
    if type(corpus_slice) is ignis.corpus.Corpus:
        corpus_slice = corpus_slice.slice_full()
    if not type(corpus_slice) is ignis.corpus.CorpusSlice:
        raise ValueError(
            "Ignis models must be instantiated with Corpus or CorpusSlice instances."
        )

    if model_type == "tp_lda":
        model = ignis.models.LDAModel(corpus_slice, model_options)
        model.train()
        aurum = ignis.aurum.Aurum(model)
    elif model_type == "tp_lda_wp":
        # Tomotopy LDA model with word priors
        # (contrib.: C. Ow)
        if isinstance(pre_model, ignis.models.lda.LDAModel):
            model = pre_model
        else:
            raise ValueError(
                "Ignis models with word priors must be pre-instantiated "
                "`ignis.models.lda.LDAModel` instances."
            )
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


def compare_topic_count_coherence(
    corpus_slice,
    model_type="tp_lda",
    model_options=None,
    coherence="c_npmi",
    top_n=30,
    start_k=3,
    end_k=10,
    iterations=150,
    verbose=True,
):
    """
    Lightly trains models with various topic counts and reports the resultant coherence
    scores.

    These scores can be used as a heuristic for choosing the number of topics to use
    for full training (e.g., via `suggest_num_topics()`).

    Parameters
    ----------
    corpus_slice: ignis.corpus.Corpus or ignis.corpus.CorpusSlice
        The `ignis.corpus.CorpusSlice` to perform the topic modelling over.  If a
        `ignis.corpus.Corpus` is passed instead, a `ignis.corpus.CorpusSlice`
        containing all of its `ignis.corpus.Document` objects will be created.
    model_type: {"tp_lda", "tp_hdp"}
        Type of model to train; corresponds to the model type listed in the relevant
        `ignis.models` class.
    model_options: dict, optional
        Dictionary of options that will be passed to the relevant `ignis.models`
        model constructor.
    coherence: {"c_npmi", "c_v", "u_mass", "c_uci"}, optional
        Coherence measure to calculate. `"c_npmi"` by default.
    top_n: int, optional
        Number of top tokens to extract from each topic when measuring coherence.
        The default of 30 matches the number of tokens shown per topic by pyLDAvis.
    start_k: int, optional
        Minimum topic count to consider.
    end_k: int, optional
        Maximum topic count to consider.
    iterations: int, optional
        Number of iterations to train each candidate model for.
    verbose: bool, optional
        Whether or not to show interim training progress.

    Returns
    -------
    iterable of tuple
        A list of tuples (`topic count`, `coherence score`) for all the topic counts
        in the test range.
    """
    if type(corpus_slice) is ignis.corpus.Corpus:
        corpus_slice = corpus_slice.slice_full()
    if not type(corpus_slice) is ignis.corpus.CorpusSlice:
        raise ValueError(
            "Ignis models must be instantiated with Corpus or CorpusSlice instances."
        )

    if model_options is None:
        model_options = {}

    progress_bar = None
    if verbose:
        total_models = end_k - start_k + 1
        print(
            f"Training {total_models} mini-models to suggest a suitable number of "
            f"topics between {start_k} and {end_k}...\n"
            f"({len(corpus_slice)} documents, {iterations} iterations each, "
            f"coherence metric: '{coherence}')"
        )
        progress_bar = tqdm(total=total_models * iterations, miniters=1)

    results = []
    candidate_counts = range(start_k, end_k + 1)
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
                    ),
                )
            )

        if verbose:
            progress_bar.update(iterations)
            # To allow the tqdm bar to update, if in a Jupyter notebook
            time.sleep(0.01)

    if verbose:
        progress_bar.close()
    return results


def suggest_num_topics(*args, verbose=True, **kwargs):
    """
    Convenience function for running `compare_topic_count_coherence()` and directly
    reporting the topic count with the highest coherence found.

    Parameters
    ----------
    verbose: bool, optional
        Whether or not to print the details of the best topic count.
    *args, **kwargs
        Passed on to `compare_topic_count_coherence()`.

    Returns
    -------
    int
        The suggested topic count.
    """
    results = compare_topic_count_coherence(*args, verbose=verbose, **kwargs)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    best = results[0]

    if verbose:
        print(f"Suggested topic count: {best[0]}\t" f"Coherence: {best[1]}")

    return best[0]
