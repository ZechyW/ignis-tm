import pathlib
import shutil
import threading
import time
import warnings

try:
    # Don't depend fully on a Jupyter environment, in case the user wants to create
    # headless visualisations
    from IPython.core.display import display, HTML
except ModuleNotFoundError:
    pass

# We monkey patch pyLDAvis to optimise various pandas calculations below
from joblib import Parallel, delayed

import ignis.util

pyLDAvis = ignis.util.LazyLoader("pyLDAvis")
np = ignis.util.LazyLoader("numpy")
pd = ignis.util.LazyLoader("pandas")

# noinspection PyProtectedMember
_prepare = pyLDAvis._prepare


def show_visualisation(vis_data):
    """
    Display the pyLDAvis visualisation for the given data; assumes a Jupyter notebook
    environment.

    Parameters
    ----------
    vis_data

    Returns
    -------
    IPython.display.HTML
    """
    # CSS styles for displaying pyLDAvis visualisations nicely
    # - Resize to fit visualisations without causing other cells to overflow
    jupyter_styles = """
    <style>
        /* These have to be marked important to override pyLDAvis default styles */
        #notebook-container {
            width: 1370px !important;
        }
    
        div.output_area {
            width: unset !important;
        }
        
        div.output_html.rendered_html {
            max-height: unset;
        }
    </style>
    """
    # noinspection PyTypeChecker
    display(HTML(jupyter_styles))

    with warnings.catch_warnings():
        try:
            import IPython.utils.shimmodule

            # Let's pretend pyLDAvis isn't a few generations behind ¯\_(ツ)_/¯
            # (Waiting for upstream 2.1.4)
            warnings.simplefilter(
                "ignore", category=IPython.utils.shimmodule.ShimWarning
            )
        except ModuleNotFoundError:
            pass

        return pyLDAvis.display(vis_data, local=True)


def export_visualisation(vis_data, folder):
    """
    Exports a pyLDAvis visualisation of `vis_data` as a HTML file in the given folder

    Attempts to copy the stock visualisation sources (js/css/etc) over rather than
    assuming Internet access is available.

    Parameters
    ----------
    vis_data: pyLDAvis.PreparedData
    folder
    """
    folder = pathlib.Path(folder)
    folder.mkdir(exist_ok=True)

    # Copy the pyLDAvis sources
    sources_folder = folder / "src"
    sources_folder.mkdir(exist_ok=True)

    d3_src = pathlib.Path(pyLDAvis.urls.D3_LOCAL)
    ldavis_src = pathlib.Path(pyLDAvis.urls.LDAVIS_LOCAL)
    ldavis_css = pathlib.Path(pyLDAvis.urls.LDAVIS_CSS_LOCAL)
    for src in [d3_src, ldavis_src, ldavis_css]:
        shutil.copy2(src, sources_folder)

    # These urls are relative to the HTML file
    local_urls = {
        "d3_url": "src/" + d3_src.name,
        "ldavis_url": "src/" + ldavis_src.name,
        "ldavis_css_url": "src/" + ldavis_css.name,
    }

    # pyLDAvis expects strings or file objects
    output = str(folder / "visualisation.html")

    pyLDAvis.save_html(vis_data, output, **local_urls)


def prepare_data(
    model,
    mds="pcoa",
    lambda_step=0.1,
    sort_topics=False,
    verbose=False,
    use_optimised=True,
    **other_options,
):
    """
    Provides a simple interface for preparing the data for visualisation with pyLDAvis

    Parameters
    ----------
    model: tp.LDAModel
        A trained Tomotopy model
    verbose: bool, optional
        Whether or not to print verbose progress messages
    use_optimised: bool, optional
        Whether to use our optimised, monkey-patched version of the pyLDAvis prepare
        function; will use the original one otherwise.

    mds
    lambda_step
    sort_topics
        These other keyword arguments are pyLDAvis-specific options
        (See pyLDAvis docs for details)
    """
    origin_time = time.perf_counter()

    # Convert tomotopy model data to pyLDAvis format
    # ----------------------------------------------

    model_data = {
        "topic_term_dists": [model.get_topic_word_dist(k) for k in range(model.k)],
        "doc_topic_dists": [
            model.docs[n].get_topic_dist() for n in range(len(model.docs))
        ],
        "doc_lengths": [len(model.docs[n].words) for n in range(len(model.docs))],
        "vocab": model.vocabs,
        "term_frequency": model.vocab_freq,
    }

    # Since we are doing the actual calculations in a separate thread, we collect the
    # options here to pass them through more neatly
    # (Is there a better way to handle this?)
    options = dict(
        {"mds": mds, "lambda_step": lambda_step, "sort_topics": sort_topics},
        **other_options,
    )

    if verbose:
        print("Preparing LDA visualisation...", flush=True, end="")

    results = [None]

    if use_optimised:
        # We probably don't need to start up a separate thread if we are using our
        # monkey-patched optimised prepare function
        _prepare_vis(model_data, options, use_optimised, results)
    else:
        t = threading.Thread(
            target=_prepare_vis, args=(model_data, options, use_optimised, results)
        )
        t.start()

        progress_countdown = 1.0

        while t.is_alive():
            time.sleep(0.1)
            progress_countdown -= 0.1
            if progress_countdown <= 0:
                if verbose:
                    print(" .", flush=True, end="")
                progress_countdown = 1

    vis_data = results[0]
    elapsed = time.perf_counter() - origin_time

    if verbose:
        print(f" Done. ({elapsed:.3f}s)")

    return vis_data


def _prepare_vis(model_data, options, use_optimised, results):
    """
    Helper function to call the `pyLDAvis.prepare` method in a separate thread so
    that we can monitor progress.

    Parameters
    ----------
    model_data
        Raw model data in the format expected by pyLDAvis.
    options: dict
        PyLDAvis options
    use_optimised: bool
        Whether to use our optimised, monkey-patched version of the pyLDAvis prepare
        function; will use the original one otherwise.
    results: iterable
        Single element list to be passed in by reference -- The prepared data will be
        stored here.
    """
    if use_optimised:
        vis_data = _fast_prepare(**model_data, **options)
    else:
        vis_data = pyLDAvis.prepare(**model_data, **options)
    results[0] = vis_data


def _fast_prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    R=30,
    lambda_step=0.01,
    mds=pyLDAvis.js_PCoA,
    n_jobs=-1,
    plot_opts=None,
    sort_topics=True,
    skip_validate=False,
):
    """
    Helper function that runs optimised versions of the pyLDAvis functions to reduce
    runtime complexity, especially for later versions of `pandas` (e.g., > 1.0.0).

    Much of the code will be taken directly from `_prepare.py` in the pyLDAvis
    package (v2.1.2), with modifications noted in the comments.

    Parameters
    ----------
    topic_term_dists : array-like, shape (`n_topics`, `n_terms`)
        Matrix of topic-term probabilities. Where `n_terms` is `len(vocab)`.
    doc_topic_dists : array-like, shape (`n_docs`, `n_topics`)
        Matrix of document-topic probabilities.
    doc_lengths : array-like, shape `n_docs`
        The length of each document, i.e. the number of words in each document.
        The order of the numbers should be consistent with the ordering of the
        docs in `doc_topic_dists`.
    vocab : array-like, shape `n_terms`
        List of all the words in the corpus used to train the model.
    term_frequency : array-like, shape `n_terms`
        The count of each particular term over the entire corpus. The ordering
        of these counts should correspond with `vocab` and `topic_term_dists`.
    R : int
        The number of terms to display in the barcharts of the visualization.
        Default is 30. Recommended to be roughly between 10 and 50.
    lambda_step : float, between 0 and 1
        Determines the interstep distance in the grid of lambda values over
        which to iterate when computing relevance.
        Default is 0.01. Recommended to be between 0.01 and 0.1.
    mds : function or a string representation of function
        A function that takes `topic_term_dists` as an input and outputs a
        `n_topics` by `2`  distance matrix. The output approximates the distance
        between topics. See :func:`js_PCoA` for details on the default function.
        A string representation currently accepts `pcoa` (or upper case variant),
        `mmds` (or upper case variant) and `tsne` (or upper case variant),
        if `sklearn` package is installed for the latter two.
    n_jobs : int
        The number of cores to be used to do the computations. The regular
        joblib conventions are followed so `-1`, which is the default, will
        use all cores.
    plot_opts : dict, with keys 'xlab' and `ylab`
        Dictionary of plotting options, right now only used for the axis labels.
    sort_topics : sort topics by topic proportion (percentage of tokens covered). Set
        to false to keep original topic order.

    skip_validate: bool, optional
        If set, will ignore validation errors (e.g., those caused by numerical
        instability).  Use with caution.

    Returns
    -------
    pyLDAvis.PreparedData
    """
    # ZW: Make default `plot_opts` immutable
    if plot_opts is None:
        plot_opts = {"xlab": "PC1", "ylab": "PC2"}

    # parse mds
    # ZW: if isinstance(mds, basestring):
    if isinstance(mds, str):
        mds = mds.lower()
        if mds == "pcoa":
            mds = _prepare.js_PCoA
        elif mds in ("mmds", "tsne"):
            if _prepare.sklearn_present:
                mds_opts = {"mmds": _prepare.js_MMDS, "tsne": _prepare.js_TSNE}
                mds = mds_opts[mds]
            else:
                _prepare.logging.warning("sklearn not present, switch to PCoA")
                mds = _prepare.js_PCoA
        else:
            _prepare.logging.warning("Unknown mds `%s`, switch to PCoA" % mds)
            mds = _prepare.js_PCoA

    # ZW: Pandas is column-oriented, but the tomotopy `topic_term_dists` is naturally
    # arranged by row.  We can save a bunch of data prep runtime by pre-building the
    # DataFrame instead of passing an array of rows to the constructor.
    # (There are way more columns, which otherwise have to be re-constructed from the
    # array, than there are rows -- There are as many columns as terms, but only as
    # many rows as topics)
    topic_term_dist_cols = [
        pd.Series(topic_term_dist, dtype="float64")
        for topic_term_dist in topic_term_dists
    ]
    topic_term_dists = pd.concat(topic_term_dist_cols, axis=1).T

    topic_term_dists = _df_with_names(topic_term_dists, "topic", "term")
    doc_topic_dists = _df_with_names(doc_topic_dists, "doc", "topic")
    term_frequency = _series_with_name(term_frequency, "term_frequency")
    doc_lengths = _series_with_name(doc_lengths, "doc_length")
    vocab = _series_with_name(vocab, "vocab")
    if not skip_validate:
        _prepare._input_validate(
            topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
        )
    R = min(R, len(vocab))

    topic_freq = doc_topic_dists.mul(doc_lengths, axis="index").sum()
    # ZW: topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    # topic_freq       = np.dot(doc_topic_dists.T, doc_lengths)
    if sort_topics:
        topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
    else:
        topic_proportion = topic_freq / topic_freq.sum()

    topic_order = topic_proportion.index
    # reorder all data based on new ordering of topics
    topic_freq = topic_freq[topic_order]
    topic_term_dists = topic_term_dists.iloc[topic_order]
    doc_topic_dists = doc_topic_dists[topic_order]

    # token counts for each term-topic combination (widths of red bars)
    term_topic_freq = (topic_term_dists.T * topic_freq).T
    # Quick fix for red bar width bug.  We calculate the
    # term frequencies internally, using the topic term distributions and the
    # topic frequencies, rather than using the user-supplied term frequencies.
    # For a detailed discussion, see: https://github.com/cpsievert/LDAvis/pull/41
    term_frequency = np.sum(term_topic_freq, axis=0)

    topic_info = _topic_info(
        topic_term_dists,
        topic_proportion,
        term_frequency,
        term_topic_freq,
        vocab,
        lambda_step,
        R,
        n_jobs,
    )
    token_table = _prepare._token_table(
        topic_info, term_topic_freq, vocab, term_frequency
    )
    topic_coordinates = _prepare._topic_coordinates(
        mds, topic_term_dists, topic_proportion
    )
    client_topic_order = [x + 1 for x in topic_order]

    return _prepare.PreparedData(
        topic_coordinates,
        topic_info,
        token_table,
        R,
        lambda_step,
        plot_opts,
        client_topic_order,
    )


def _df_with_names(data, index_name, columns_name):
    """ From `pyLDAvis._prepare.py` """
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        # ZW: `from_records()` might be slightly more performant?
        df = pd.DataFrame.from_records(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    """ From `pyLDAvis._prepare.py` """
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def _topic_info(
    topic_term_dists,
    topic_proportion,
    term_frequency,
    term_topic_freq,
    vocab,
    lambda_step,
    R,
    n_jobs,
):
    """ From `pyLDAvis._prepare.py`, optimised """
    # marginal distribution over terms (width of blue bars)
    term_proportion = term_frequency / term_frequency.sum()

    # compute the distinctiveness and saliency of the terms:
    # this determines the R terms that are displayed when no topic is selected
    tt_sum = topic_term_dists.sum()
    topic_given_term = pd.eval("topic_term_dists / tt_sum")
    # ZW: topic_given_term = topic_term_dists / topic_term_dists.sum()
    log_1 = np.log(pd.eval("(topic_given_term.T / topic_proportion)"))
    kernel = pd.eval("topic_given_term * log_1.T")
    # ZW: kernel = topic_given_term * np.log((topic_given_term.T / topic_proportion).T)
    distinctiveness = kernel.sum()
    saliency = term_proportion * distinctiveness

    # Order the terms for the "default" view by decreasing saliency:
    default_term_info = (
        pd.DataFrame(
            {
                "saliency": saliency,
                "Term": vocab,
                "Freq": term_frequency,
                "Total": term_frequency,
                "Category": "Default",
            }
        )
        .sort_values(by="saliency", ascending=False)
        .head(R)
        .drop("saliency", 1)
    )
    # Rounding Freq and Total to integer values to match LDAvis code:
    default_term_info["Freq"] = np.floor(default_term_info["Freq"])
    default_term_info["Total"] = np.floor(default_term_info["Total"])
    ranks = np.arange(R, 0, -1)
    default_term_info["logprob"] = default_term_info["loglift"] = ranks

    # compute relevance and top terms for each topic
    log_lift = np.log(pd.eval("topic_term_dists / term_proportion")).astype("float64")
    # ZW: log_lift = np.log(topic_term_dists / term_proportion)
    log_ttd = np.log(topic_term_dists).astype("float64")
    lambda_seq = np.arange(0, 1 + lambda_step, lambda_step)

    def topic_top_term_df(tup):
        new_topic_id, (original_topic_id, topic_terms) = tup
        term_ix = topic_terms.unique()
        # ZW: Changed order below to match `default_term_info`
        return pd.DataFrame(
            {
                "Term": vocab[term_ix],
                "Freq": term_topic_freq.loc[original_topic_id, term_ix],
                "Total": term_frequency[term_ix],
                "Category": "Topic%d" % new_topic_id,
                "logprob": log_ttd.loc[original_topic_id, term_ix].round(4),
                "loglift": log_lift.loc[original_topic_id, term_ix].round(4),
            }
        )

    top_terms = pd.concat(
        # ZW: Parallel(n_jobs=n_jobs)(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_find_relevance_chunks)(log_ttd, log_lift, R, ls)
            for ls in _prepare._job_chunks(lambda_seq, n_jobs)
        )
    )
    topic_dfs = map(topic_top_term_df, enumerate(top_terms.T.iterrows(), 1))
    return pd.concat([default_term_info] + list(topic_dfs))


def _find_relevance(log_ttd, log_lift, R, lambda_):
    """ From `pyLDAvis._prepare.py`, optimised """
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return relevance.T.apply(lambda topic: topic.nlargest(R).index)
    # ZW: return relevance.T.apply(lambda s: s.sort_values(
    # ascending=False).index).head(R)


def _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq):
    """ From `pyLDAvis._prepare.py` """
    return pd.concat([_find_relevance(log_ttd, log_lift, R, l) for l in lambda_seq])
