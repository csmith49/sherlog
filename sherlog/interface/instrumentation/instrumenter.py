from json import dumps
from typing import Optional, Mapping, Any
from ..console import console

class Instrumenter:
    """Instrumenters record key-value pairs and emit them to a log."""

    def __init__(self, filepath : Optional[str] = None, context : Optional[Mapping[str, Any]] = None):
        """Construct an instrumenter that writes to `filepath`."""

        self._filepath = filepath
        self._context = context if context else {}

        # message in-progress
        self._message = {}

    def set(self, **kwargs):
        """Set key-value pairs in the current message."""

        self._message |= kwargs

    @property
    def message(self):
        """The full current message, context included."""

        return self._message | self._context

    def flush(self):
        """Write the current message to file and start a new message."""

        if self._filepath:
            with open(self._filepath, 'a') as f:
                f.write(dumps(self.message))
        else:
            console.print(self.message)

        self._message = {}

    def write(self, **kwargs):
        self.set(**kwargs)
        self.flush()

    # more natural interface for self.set
    def __setitem__(self, key : str, value : Any):
        self.set(**{key : value})

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.flush()