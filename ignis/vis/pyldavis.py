import threading
import time

import pyLDAvis

default_options = {
    "mds": "pcoa",
    "lambda_step": 0.1,
    "sort_topics": False,
    "verbose": False,
}


def prepare_data(model, options=None):
    """
    Provides a simple interface for preparing the data for visualisation with pyLDAvis

    Parameters
    ----------
    model: tp.LDAModel
        A trained Tomotopy model
    options: dict, optional
        Vis-specific options (See pyLDAvis docs for details)
    """
    if options is None:
        options = {}
    options = dict(default_options, **options)

    verbose = options["verbose"]

    # Prepare the visualisation data
    model_data = {
        "topic_term_dists": [model.get_topic_word_dist(k) for k in range(model.k)],
        "doc_topic_dists": [
            model.docs[n].get_topic_dist() for n in range(len(model.docs))
        ],
        "doc_lengths": [len(model.docs[n].words) for n in range(len(model.docs))],
        "vocab": model.vocabs,
        "term_frequency": model.vocab_freq,
    }

    if verbose:
        print("Preparing LDA visualisation", flush=True, end="")

    results = [None]
    t = threading.Thread(target=_prepare_vis, args=(model_data, options, results))
    t.start()

    progress_countdown = 1.0

    while t.is_alive():
        time.sleep(0.1)
        progress_countdown -= 0.1
        if progress_countdown <= 0:
            if verbose:
                print(" .", flush=True, end="")
            progress_countdown = 1

    if verbose:
        print(" Done.")

    vis_data = results[0]
    return vis_data


def _prepare_vis(model_data, options, results):
    """
    Helper function to call the `pyLDAvis.prepare` method in a separate thread so
    that we can monitor progress.

    Parameters
    ----------
    model_data
        Raw model data in the format expected by pyLDAvis.
    options: dict
        PyLDAvis options
    results: iterable
        Single element list to be passed in by reference -- The prepared data will be
        stored here.
    """
    mds = options["mds"]
    lambda_step = options["lambda_step"]
    sort_topics = options["sort_topics"]

    vis_data = pyLDAvis.prepare(
        **model_data, mds=mds, lambda_step=lambda_step, sort_topics=sort_topics
    )
    results[0] = vis_data
