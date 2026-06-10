"""Logging helpers for the align runtime."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

try:  # Optional dependency for rich progress bars.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


_VERBOSITY = {
    "quiet": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def configure_logging(verbosity: str = "info") -> logging.Logger:
    level = _VERBOSITY.get(verbosity.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("align")
    logger.setLevel(level)
    return logger


@dataclass
class ProgressReporter:
    total: int
    initial: int = 0

    def __post_init__(self) -> None:
        self.count = self.initial
        self._bar = None
        self.uses_tqdm = False
        if tqdm is not None:
            self._bar = tqdm(
                total=self.total,
                initial=self.initial,
                dynamic_ncols=True,
                leave=False,
                position=0,
            )
            self.uses_tqdm = True

    def update(self, label: str | None = None, advance: int = 1) -> None:
        self.count += advance
        if self._bar is not None:
            if label:
                self._bar.set_description_str(label)
            self._bar.update(advance)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


@contextmanager
def progress_bar(total: int, initial: int = 0) -> Iterator[ProgressReporter]:
    reporter = ProgressReporter(total=total, initial=initial)
    try:
        yield reporter
    finally:
        reporter.close()
