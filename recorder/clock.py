import time


class RecorderClock:
    """Shared monotonic host clock for all recorder components."""

    def __init__(self) -> None:
        self._start_ns: int | None = None

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    @property
    def is_started(self) -> bool:
        return self._start_ns is not None

    @property
    def start_ns(self) -> int:
        if self._start_ns is None:
            raise RuntimeError("RecorderClock has not been started.")
        return self._start_ns

    def now_ns(self) -> int:
        if self._start_ns is None:
            raise RuntimeError("RecorderClock has not been started.")
        return time.perf_counter_ns() - self._start_ns

    def now_s(self) -> float:
        return self.now_ns() / 1_000_000_000.0
