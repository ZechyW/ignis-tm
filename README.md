# Ignis: Iterative Topic Modelling Platform

N.B.: As at version 0.8.1, Tomotopy seems to rely on `PYTHONHASHSEED` being set in order to consistently reproduce results (together with setting the actual model seed.)

If using a Conda environment, this can be done with:
```
conda env config vars set PYTHONHASHSEED=<seed>
```

For direct invocation:
```
PYTHONHASHSEED=<seed> python script.py
```

For Jupyter notebooks in a non-Conda environment, edit the Jupyter `kernel.json` to add an appropriate `env` key.
