import threading
import time

import pyLDAvis


def prepare_data(model, mds="pcoa", lambda_step=0.1, sort_topics=False, verbose=False):
    """
    Provides a simple interface for preparing the data for visualisation with pyLDAvis

    Parameters
    ----------
    model: tp.LDAModel
        A trained Tomotopy model
    verbose: bool, optional
        Whether or not to print verbose progress messages

    mds
    lambda_step
    sort_topics
        These other keyword arguments are pyLDAvis-specific options
        (See pyLDAvis docs for details)
    """
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
    options = {"mds": mds, "lambda_step": lambda_step, "sort_topics": sort_topics}

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
    vis_data = pyLDAvis.prepare(**model_data, **options)
    results[0] = vis_data
