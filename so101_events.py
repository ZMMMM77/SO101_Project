"""Optional observers for hardware I/O used by the ROS 2 entry points."""

from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar("so101_observer", default=None)


@contextmanager
def observe_with(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def report_position(position, *, command=False):
    observer = _observer.get()
    if observer is not None:
        observer(position, command=command)
    return position
