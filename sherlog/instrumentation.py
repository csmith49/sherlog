from time import perf_counter as current_time
from json import dumps
from hashids import Hashids
from random import randint
import torch

# set up the hashid conversion for seed generation
hashids = Hashids()

class Timer:
    __slots__ = ("_start_time", "_stop_time")

    def __init__(self):
        '''Constructs (and starts) a timer.

        Uses the highest-precision clock available (via `time.perf_counter`) to record elapsed time in seconds.
        '''
        self._start_time = None
        self._stop_time = None

        self.start()

    def start(self):
        '''Starts the timer. Returns the instance.

        Returns
        -------
        Timer
        '''
        self._start_time = current_time()
        return self

    def stop(self):
        '''Stops the timer. Returns the instance.

        Returns
        -------
        Timer
        '''
        self._stop_time = current_time()
        return self

    def reset(self):
        '''Resets and restarts the timer. Returns the instance.

        Returns
        -------
        Timer
        '''
        self._stop_time = None
        return self

    @property
    def elapsed(self):
        '''If the timer has not been stopped, returns the elapsed time (in fractional seconds) since the timer was last started. Otherwise, returns the time elapsed between when the timer was last started and stopped.

        Returns
        -------
        float
        '''
        if self._stop_time is None:
            return current_time() - self._start_time
        else:
            return self._stop_time - self._start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

class Instrumenter:
    def __init__(self, filepath, context=None):
        '''An instrumenter that writes observations to `filepath` in JSONL format.

        Parameters
        ----------
        filepath : str

        context : Optional[Dict[str, Any]]
        '''
        self._filepath = filepath
        if context:
            self._context = context
        else:
            self._context = {}
        self._cache = []

    def emit(self, **kwargs):
        entry = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                entry[k] = v.item()
            else:
                entry[k] = v
        entry.update(self._context)
        self._cache.append(entry)

    def flush(self):
        with open(self._filepath, 'a') as f:
            for entry in self._cache:
                message = dumps(entry, skipkeys=True)
                f.write(f"{message}\n")
        self._cache = []

    def __enter__(self): return self

    def __exit__(self, *args): self.flush()

def seed():
    return hashids.encode(randint(0, 100000))
