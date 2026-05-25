from __future__ import annotations

from contextlib import contextmanager
import sys
import time
from typing import Iterable, Iterator, Literal

ProgressMode = Literal["none", "text", "tqdm", "auto"]


def _in_notebook() -> bool:
    try:
        get_ipython = __import__("IPython").get_ipython
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    return shell.__class__.__name__ in {"ZMQInteractiveShell", "Shell"}


class ProgressReporter:
    """Small optional progress helper for CLI and notebook runs."""

    def __init__(self, mode: ProgressMode | bool | None = None, *, stream=None) -> None:
        if mode is True:
            mode = "auto"
        elif mode is False or mode is None:
            mode = "none"
        if mode not in {"none", "text", "tqdm", "auto"}:
            raise ValueError("progress must be one of None, False, True, 'none', 'text', 'tqdm', or 'auto'.")

        if mode == "auto":
            mode = "tqdm" if _in_notebook() else "text"
        self.mode: ProgressMode = mode
        self.stream = stream or sys.stderr
        self._tqdm = None
        if self.mode == "tqdm":
            try:
                from tqdm.auto import tqdm

                self._tqdm = tqdm
            except Exception:
                self.mode = "text"

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def _write(self, message: str) -> None:
        if self.mode == "none":
            return
        if self._tqdm is not None:
            self._tqdm.write(message)
        else:
            print(message, file=self.stream, flush=True)

    @contextmanager
    def phase(self, name: str, detail: str | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        label = f"{name}: {detail}" if detail else name
        start = time.perf_counter()
        self._write(f"[denoistpy] {label} ...")
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - start
            self._write(f"[denoistpy] {label} failed after {elapsed:.1f}s")
            raise
        elapsed = time.perf_counter() - start
        self._write(f"[denoistpy] {label} done in {elapsed:.1f}s")

    def iter_batches(
        self,
        iterable: Iterable[int],
        *,
        total: int,
        label: str,
        batch_size: int,
        n_items: int,
    ) -> Iterator[int]:
        if self.mode == "none":
            yield from iterable
            return

        if self._tqdm is not None:
            yield from self._tqdm(
                iterable,
                total=total,
                desc=label,
                unit="batch",
                leave=False,
            )
            return

        report_every = max(1, total // 10)
        for idx, start in enumerate(iterable, start=1):
            if idx == 1 or idx == total or idx % report_every == 0:
                stop = min(start + batch_size, n_items)
                self._write(f"[denoistpy] {label}: batch {idx}/{total} cells {start}:{stop}")
            yield start
