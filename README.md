# Ignis: Iterative Topic Modelling Platform

## Dependencies

General:

```
conda install python=3.7.6 tqdm jupyter
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
conda install pandas=0.23.4 pyldavis
```

For ease of use:
```
conda install jupyter_contrib_nbextensions black
```

## Tomotopy
N.B.: As at version 0.8.1, Tomotopy sometimes seems to rely on `PYTHONHASHSEED` being set in order to consistently reproduce results (together with setting the actual model seed), although this behaviour is not always reproducible.  To be safe, `PYTHONHASHSEED` should be explicitly set where necessary.

P.S.: This may have been an artifact of Windows Prefetching; needs further testing.

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

For zero-indexing support, clone `pyLDAvis` directly from the Github repo, build the wheel, and install it.

```
conda install numpy scipy pandas=0.23.4 numexpr future funcy
conda install atomicwrites more-itertools packaging pluggy py pyparsing pytest
git clone https://github.com/bmabey/pyLDAvis.git
cd pyLDAvis
git checkout 2da07084e9df7d51a0daf240db1a64022e3023a5
python setup.py bdist_wheel
cd dist
pip install pyLDAvis-2.1.3-py2.py3-none-any.whl
```

In particular, the older versions of Pandas (e.g., v0.23.4) seem to generate the visualisation data much more quickly than the latest versions, for the latest version of pyLDAvis.