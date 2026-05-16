from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.tree import Tree


class NullReporter:
    enabled = False

    @contextmanager
    def phase(self, title: str, total_steps: int | None = None) -> Iterator[None]:
        _ = (title, total_steps)
        yield

    def status(self, message: str) -> None:
        _ = message

    def posterior_summary(self, frame: pl.DataFrame) -> None:
        _ = frame

    def tree(self, title: str, items: list[str]) -> None:
        _ = (title, items)

    def save(self, artifact_dir: Path) -> None:
        _ = artifact_dir


class RichReporter:
    enabled = True

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.console = Console(record=True, quiet=quiet)
        self._lines: list[str] = []

    @contextmanager
    def phase(self, title: str, total_steps: int | None = None) -> Iterator[None]:
        suffix = f" ({total_steps} steps)" if total_steps is not None else ""
        self.status(f"START {title}{suffix}")
        try:
            yield
        except Exception as exc:
            self.status(f"FAIL {title}: {exc}")
            raise
        self.status(f"DONE {title}")

    def status(self, message: str) -> None:
        self._lines.append(message)
        self.console.print(f"[cyan]{message}[/cyan]")

    def posterior_summary(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            self.status("posterior summary: no rows")
            return
        table = Table(title="Posterior summary")
        columns = frame.columns[:6]
        for column in columns:
            table.add_column(column)
        for row in frame.head(12).iter_rows(named=True):
            table.add_row(*(str(row.get(column, "")) for column in columns))
        self._lines.append(f"posterior summary rows={frame.height}")
        self.console.print(table)

    def tree(self, title: str, items: list[str]) -> None:
        tree = Tree(title)
        for item in items:
            tree.add(item)
        self._lines.append(f"{title}: {', '.join(items)}")
        self.console.print(tree)

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "inference.log").write_text(
            "\n".join(self._lines) + ("\n" if self._lines else ""),
            encoding="utf-8",
        )
        self.console.save_html(str(artifact_dir / "inference.html"), clear=False)


def get_reporter(quiet: bool | None = None) -> RichReporter | NullReporter:
    env_quiet = os.environ.get("RICH_QUIET", "").strip().lower() in {"1", "true", "yes"}
    if quiet is None:
        quiet = env_quiet
    else:
        quiet = bool(quiet) or env_quiet
    return RichReporter(quiet=quiet)
