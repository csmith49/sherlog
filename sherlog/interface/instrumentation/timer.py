from time import perf_counter as current_time

# context manager for precise timing
class Timer:
    """Timing helper that uses the highest-precision clock available (via `time.perf_counter`)."""

    def __init__(self):
        """Construct (and start) a timer."""

        self._start_time = None
        self._stop_time = None

        self.start()

    def start(self):
        """Start the timer. Returns the instance."""

        self._start_time = current_time()
        self._stop_time = None
        return self

    def stop(self):
        """Stops the timer. Returns the instance."""

        self._stop_time = current_time()
        return self

    @property
    def elapsed(self):
        """The elapsed time (in fractional seconds) since the timer was started.

        If the timer is running (i.e., has been started but not stopped), the elapsed time is `now - start`, otherewise it is `stop - start`.
        """

        if self._stop_time is None:
            return current_time() - self._start_time
        else:
            return self._stop_time - self._start_time

    # CONTEXT MANAGER INTERFACE

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()