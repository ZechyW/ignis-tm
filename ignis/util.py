import importlib


class LazyLoader:
    """
    A general class for lazy loading expensive modules.

    Parameters
    ----------
    module_name: str
        Module name to lazy load

    Examples
    --------
    tp = LazyLoader("tomotopy")
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, item):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, item)
