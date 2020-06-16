import pathlib
import shutil
import threading
import time
import warnings
import pandas as pd

import pyLDAvis
import pyLDAvis.urls
import pyLDAvis.utils


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
    origin_time = time.perf_counter()

    # Convert tomotopy model data to pyLDAvis format
    # ----------------------------------------------

    # Pandas appears to operate in a column-based fashion, but the tomotopy
    # topic_term_dists is naturally arranged by row;
    # We can save a bunch of runtime by preparing the DataFrame before sending it off
    # to pyLDAvis, since there are way more columns than there are rows
    # (There are as many columns as terms, but only as many rows as topics)

    # By default Pandas thinks the DataFrame is a single column, but it's really a
    # single row, so we Transpose
    topic_term_dists_rows = [
        pd.DataFrame(model.get_topic_word_dist(k)).T for k in range(model.k)
    ]
    topic_term_dists = pd.concat(topic_term_dists_rows, ignore_index=True)

    model_data = {
        "topic_term_dists": topic_term_dists,
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

    elapsed = time.perf_counter() - origin_time

    if verbose:
        print(f" Done. ({elapsed:.3f}s)")

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
    with warnings.catch_warnings():
        try:
            import IPython.utils.shimmodule

            # Let's pretend pyLDAvis isn't a few generations behind ¯\_(ツ)_/¯
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