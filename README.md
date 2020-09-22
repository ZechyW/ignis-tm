# Ignis: Iterative Topic Modelling Platform

Ignis is an extensible platform that provides a common interface for creating and visualising topic models.

By default, it supports creating LDA models using Tomotopy (https://bab2min.github.io/tomotopy/) and visualising them using pyLDAvis (https://github.com/bmabey/pyLDAvis), but support for other models and frameworks can be written in as necessary.

## Development Dependencies

If you intend to extend or modify the platform, an Anaconda 3 environment may be used to easily manage the project dependencies.  Note that the `conda-forge` (https://conda-forge.org/) channel may have to be enabled for some of these dependencies to be installed.

General:

```
conda install python=3.7 tqdm jupyter
```

Corpus-prep Demo:

```
conda install gensim nltk
python -m nltk.downloader stopwords
```

Tomotopy:

```
conda install py-cpuinfo numpy
pip install tomotopy
```

pyLDAvis:

```
conda install pandas pyldavis
```

For ease of use:
```
conda install jupyter_contrib_nbextensions black
```

Documentation:
```
conda install sphinx sphinx_rtd_theme
pip install m2r2
cd docs
sphinx-build -b html . _build
```

## Indeterminacy
N.B.: Some of the dependencies (e.g., Tomotopy, Gensim) sometimes seem to rely on `PYTHONHASHSEED` being set in order to consistently reproduce results (together with setting the actual random seed), although this behaviour is not always reproducible.  This behaviour may have been fixed with the upstream release of Tomotopy v0.9.1., but to be safe, `PYTHONHASHSEED` should be explicitly set where necessary.  

If using a Conda environment, this can be done with:
```
conda env config vars set PYTHONHASHSEED=<seed>
```

For direct invocation:
```
PYTHONHASHSEED=<seed> python script.py
```

For Jupyter notebooks in a non-Conda environment, edit the Jupyter `kernel.json` to add an appropriate `env` key.

## pyLDAvis

The older versions of Pandas (<0.24.0a) pinned by the default distribution of pyLDAvis generate the visualisation data much more quickly than newer versions of Pandas.  Ignis comes with a built-in monkey-patched version of the pyLDAvis `.prepare()` function that works better with these newer versions, and uses it by default.
