"""Logging helpers for the align runtime."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from tqdm.auto import tqdm

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


class ProgressReporter:
    """Progress bar with labeled updates."""

    def __init__(self, total: int, initial: int = 0) -> None:
        self._bar = tqdm(
            total=total,
            initial=initial,
            dynamic_ncols=True,
            leave=False,
            position=0,
        )

    def update(self, label: str | None = None, advance: int = 1) -> None:
        if label:
            self._bar.set_description_str(label)
        self._bar.update(advance)

    def close(self) -> None:
        self._bar.close()


@contextmanager
def progress_bar(total: int, initial: int = 0) -> Iterator[ProgressReporter]:
    reporter = ProgressReporter(total=total, initial=initial)
    try:
        yield reporter
    finally:
        reporter.close()
