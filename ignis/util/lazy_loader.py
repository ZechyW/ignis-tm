"""
A general class for lazy loading expensive modules.
"""

import importlib


class LazyLoader:
    """
    A general class for lazy loading expensive modules.

    Modules will only be fully loaded when any of their members are first accessed.

    Parameters
    ----------
    module_name: str
        Name of the module to lazy load

    Examples
    --------
    >>> tp = LazyLoader("tomotopy")
    >>> vars(tp)
    {'module_name': 'tomotopy', '_module': None}
    >>> tp.__version__
    '0.10.1'
    >>> tp._module is None
    False
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, item):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, item)
