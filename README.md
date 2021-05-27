# Ignis: Iterative Topic Modelling Platform

`ignis` is an extensible platform that provides a common interface for creating and visualising topic models.

By default, it supports creating LDA models using Tomotopy (https://bab2min.github.io/tomotopy/) and visualising them using pyLDAvis (https://github.com/bmabey/pyLDAvis), but support for other models and frameworks can be written in as necessary.

## Installation

The library package is named `ignis-tm` on PyPI, so to use it in a project, first install the `ignis-tm` package:

```shell
pip install ignis-tm
```

After installation, import and use the library as `ignis` in your code:

```python
import ignis
```

## Demonstration/Development Environment Walkthrough

A full demonstration/development environment can be easily set up using Python 3.7 and `pipenv`.

### Clone the repository

Start by cloning the repository and navigating to the root folder of the codebase:

```shell
git clone https://github.com/ZechyW/ignis-tm.git
cd ignis-tm
```

### Install the project dependencies

Install `pipenv` and use it to install the other dependencies:

```shell
pip install pipenv
pipenv install --dev
```

The `pipenv` environment can then be activated from the codebase root:

```shell
pipenv shell
```

The `pipenv` environment will always need to be activated before the demo Jupyter notebooks can be used. 

### Perform post-installation steps

The full demonstration setup includes a number of Jupyter plugins under its dev dependencies that could be useful for working with the sample notebooks.

With the demo environment activated, install and configure the plugins:

```shell
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user 
```

You can then configure the Jupyter notebook extensions directly from the web-based Jupyter UI.  In particular, see https://neuralcoder.science/Black-Jupyter/ for a guide to setting up the Code Prettify extension using `black`.  The ExecuteTime extension is also useful for tracking cell execution times.

## Other Notes

### Random seeds and indeterminacy
```text
N.B.: The behaviour described below should be fixed in Tomotopy >= 0.9.1, which uses a different random number generation scheme.  Note that models created with Tomotopy < 0.9.1 might therefore differ from newer models even if the same seed is set.
```

Some dependencies that perform non-deterministic operations (e.g., Tomotopy, Gensim) may need `PYTHONHASHSEED` to be set in order to consistently reproduce results.  To be safe, `PYTHONHASHSEED` should be explicitly set where necessary.  

If using a Conda environment, this can be done with:
```shell
conda env config vars set PYTHONHASHSEED=<seed>
```

For direct invocation:
```shell
PYTHONHASHSEED=<seed> python script.py
```

For Jupyter notebooks in a non-Conda environment, edit the Jupyter `kernel.json` to add an appropriate `env` key.

### Miscellaneous notes on dependencies

The `ipython` and `jedi` packages are pinned to specific versions in the demo `pipenv` environment to ensure their compatibility with extensions and code completion within Jupyter notebooks; unfortunately, they break with later versions due to a lack of upstream updates.
