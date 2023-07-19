"""Tools to convert results into pretty tables."""
from pathlib import Path

import numpy as np
import pandas as pd

import src.typing as ty
from src.utils import io

__all__ = ['TableFormatter']


def _default_key(file: Path) -> str:
    """Default function used to generate a row tag from a file. OVERRIDE IF NEEDED."""
    return file.parents[2].name  # File are usually: .../<MODEL_NAME>/<SEED>/results/<DATASET>.yaml


class TableFormatter:
    r"""Class to convert results into pretty tables.

    Currently supported:
        - LaTeX `booktabs`
        - Markdown

    When converting into Latex, this assumes a few custom commands:
    ```
    \definecolor{ForestGreen}{HTML}{228b22}  % xcolor tends to have option clash

    \newcommand{\mycaption}[2]{\caption[#1]{\textbf{#1.} #2}}
    \newcommand{\best}[1]{\textcolor{ForestGreen}{\textbf{#1}}}
    \newcommand{\nbest}[1]{\textcolor{blue}{\underline{#1}}}

    % `black` makes it easier to experiment with different colours.
    \newcommand{\up}{\textcolor{black}{\ensuremath{\boldsymbol{\uparrow}}}}
    \newcommand{\down}{\textcolor{black}{\ensuremath{\boldsymbol{\downarrow}}}}
    ```

    :param header: (list[str]) (m,) Header elements represented as str.
    :param labels: (list[str|list[str]]) (n,) Row names represented as str or list of str (multi-index names).
    :param body: (list[list[float]]) (n, m) Table data for each `label` and each `header`.
    :param metrics: (None|int|list[int]) (m,) Value for each col indicating if a high/low value is better. {+1, -1}.
    :param title: (None|str) Overall table name, placed in the upper-left corner.
    """
    def __init__(self,
                 header: ty.S[str],
                 labels: ty.S[ty.U[str, ty.S[str]]],
                 body: ty.S[ty.S[float]],
                 metrics: ty.N[ty.U[int, ty.S[int]]] = None,
                 title: ty.N[str] = None):
        self.header = header  # (m,)
        self.labels = labels  # (n,)
        self.body = np.array(body)  # (n, m)
        self.metrics = np.array(metrics)[None]  # (1, m)
        self.title = title or ''

        # If given a single metric type, assume it applies to all columns.
        if self.metrics.ndim == 1: self.metrics = self.metrics[None].repeat(len(header), axis=1)

        # Convert multi-index labels to single-index labels.
        if not isinstance(self.labels[0], str): self.labels = [' '.join(l) for l in self.labels]

        sh = len(self.labels), len(self.header)
        if sh != self.shape: raise ValueError(f'Shape mismatch. ({sh} vs. {self.shape})')

        if self.metrics.shape[1] != self.shape[1]:
            raise ValueError(f'Metric type mismatch. ({self.metrics.shape[1]} vs. {self.shape[1]})')

        self.best_mask, self.nbest_mask = self._get_best()

    @classmethod
    def from_files(cls,
                   files: ty.S[Path],
                   key: ty.Callable[[Path], str] = _default_key,
                   metrics: ty.N[ty.U[int, ty.S[int]]] = None):
        """Classmethod to create a table from a list of yaml files.

        :param files: (ty.S[Path]) ty.S of YAML files containing results.
        :param key: (None|Callable) Function to convert a file name into a tag for each row.
        :param metrics: (None|int|list[int]) Value for each col indicating whether a high/low value is better. {+1, -1}.
        :return:
        """
        assert len(files), 'Must provide files to create table.'
        return cls(
            header=list(io.load_yaml(files[0])),
            labels=list(map(key, files)),
            body=[list(io.load_yaml(f).values()) for f in files],
            metrics=metrics,
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame, metrics: ty.N[ty.U[int, ty.S[int]]] = None):
        """Classmethod to create a table from a `DataFrame`.

        :param df: (pd.DataFrame) Pandas dataframe to create the table.
        :param metrics: (ty.N[ty.S[int]]) Value for each col indicating if a high/low value is better. {+1, -1}.
        :return:
        """
        return cls(header=df.columns, labels=df.index, body=df.to_numpy(), metrics=metrics, title=df.columns.name)

    @classmethod
    def from_dict(cls, data: dict[str, float], metrics: ty.N[ty.U[int, ty.S[int]]] = None):
        """Classmethod to generate a table from a single dict."""
        return cls(header=list(data), labels=['Values'], body=np.array(list(data.values()))[None], metrics=metrics)

    def __str__(self) -> str:
        """Format as a Latex table using default parameters."""
        return self.to_latex()

    @property
    def shape(self) -> tuple[int, int]:
        """Table shape as (rows, cols)."""
        return self.body.shape

    @staticmethod
    def _to_latex_row(label: str, data: ty.S[str]) -> str:
        """Create a table row."""
        return f'{label} & {" & ".join(data)} \\\\ \n'

    @staticmethod
    def _to_md_row(label: str, data: ty.S[str]) -> str:
        """Create a table row."""
        return f'| **{label}** | {" | ".join(data)} | \n'

    def _get_best(self) -> tuple[ty.A, ty.A]:
        """Get binary mask indicating the `best` and `next best` performing row per column."""

        # If no metric types were given, we can't know which is best.
        if self.metrics[0, 0] is None:
            return np.zeros_like(self.body, dtype=bool), np.zeros_like(self.body, dtype=bool)

        body = self.body * self.metrics  # Order such that higher is always better.
        best = body.max(axis=0, keepdims=True)
        best_mask = np.equal(body, best)

        # If the table has only one row, there are no next-best items.
        if self.shape[0] == 1:
            return best_mask, np.zeros_like(body, dtype=bool)

        body[best_mask] = -np.inf  # Mask out the best item.
        nbest = body.max(axis=0, keepdims=True)
        nbest_mask = np.equal(body, nbest)
        nbest_mask[best_mask] = False  # Ensure best and next-best are not the same.

        return best_mask, nbest_mask

    def _get_col_width(self,
                       header: ty.S[str],
                       body: ty.A,
                       width: ty.N[ty.U[int, ty.S[int]]] = None,
                       multi_header: ty.N[ty.S[str]] = None) -> ty.S[int]:
        """Get width for each column: dynamic, fixed or specified.

         If `width = None`, the width will be automatically be determined based on the longest item in each column.
         If `width = int`, we apply a fixed width to all columns.
         If `width = list[int]`, we apply a predetermined width to each column.
         """
        multi_header = multi_header or ['']*len(header)

        if width is None:  # Dynamic width per column.
            width = np.concatenate((
                [io.lmap(len, multi_header)],
                [io.lmap(len, header)],
                np.vectorize(len)(body)
            ), axis=0).max(0)

        elif isinstance(width, int): width = [width]*self.shape[1]  # Same width for all columns.
        elif len(width) != self.shape[1]: raise ValueError('Non-matching columns.')

        return width

    def _get_header(self) -> tuple[ty.S[str], ty.S[str]]:
        """Convert header into header and potentially multi-header."""
        multi_header, header = [], []

        # Support for pandas MultiIndex, where column names are given as a tuple.
        # Treat the first index as an additional header row.
        for h in self.header:
            if isinstance(h, tuple):
                hh, h = h
                multi_header.append(hh.replace('_', ' '))
            header.append(h.replace('_', ' '))
        return multi_header, header

    def _sort(self,
              idx: int,
              labels: ty.S[str],
              body: ty.A,
              best_mask: ty.A,
              nbest_mask: ty.A,
              reverse: bool = False) -> tuple[ty.S[str], ty.A, ty.A, ty.A]:
        """Sort the table rows by a given column idx."""
        mult = self.metrics[0, idx] or -1
        mult *= (1 if reverse else -1)

        order = (mult * body[:, idx]).argsort()
        labels = [labels[i] for i in order]
        body, best_mask, nbest_mask = body[order], best_mask[order], nbest_mask[order]

        return labels, body, best_mask, nbest_mask

    def _pad(self, header, multi_header, body, width):
        ws = self._get_col_width(header, body, width=width, multi_header=multi_header)
        multi_header = [f'{d:>{w}}' for d, w in zip(multi_header, ws)]
        header = [f'{h:>{w}}' for h, w in zip(header, ws)]
        body = np.stack([np.vectorize(lambda i: f'{i:>{w}}')(col) for w, col in zip(ws, body.T)]).T

        return multi_header, header, body

    def to_latex(self,
                 caption: ty.U[str, tuple[str, str]] = 'CAPTION',
                 precision: int = 2,
                 width: ty.N[ty.U[int, ty.S[int]]] = None,
                 sort: ty.N[int] = None,
                 reverse: bool = False) -> str:
        """Create a Latex `booktabs` table.

        :param caption: (str) Table caption.
        :param precision: (int) Precision when rounding table `body`.
        :param width: (int) Row character width.
        :param sort: (ty.N[int]) Column index to sort by.
        :param reverse: (bool) If `True`, sort from worst to best.
        :return: (str) LaTeX table represented as a string.
        """
        caption = f'\\caption{{{caption}}}' if isinstance(caption, str) else \
            f'\\mycaption{{{caption[0]}}}{{%\n{caption[1]}.\n}}'
        multi_header, header = self._get_header()

        labels = [l.replace('_', ' ') for l in self.labels]  # Latex doesn't like `_` unless in math mode.
        body, best_mask, nbest_mask = self.body, self.best_mask, self.nbest_mask

        if sort is not None:
            labels, body, best_mask, nbest_mask = self._sort(sort, labels, body, best_mask, nbest_mask, reverse=reverse)

        # Add arrow pointing up or down depending on metric type.
        if self.metrics[0, 0] is not None:
            header = [h+('\\up' if m == 1 else '\\down') for h, m in zip(header, self.metrics[0])]

        body = np.vectorize(lambda i: f'{i:.{precision}f}')(body).astype('<U256')
        body[best_mask] = [f'\\best{{{i}}}' for i in body[best_mask]]
        body[nbest_mask] = [f'\\nbest{{{i}}}' for i in body[nbest_mask]]

        multi_header, header, body = self._pad(header, multi_header, body, width)

        table = (
            '% =====================================\n'
            '\\begin{table}\n'
            '\\renewcommand{\\arraystretch}{1.2}\n'
            '\\centering\n'
            '\\scriptsize\n'
            '\\addtolength{\\tabcolsep}{-0.6em}\n'
            '' + caption + '\n'
            '\\begin{tabular}{@{}' + 'l'*(len(header)+1) + '@{}}\n'
            '\\toprule\n'
        )

        n = max(map(len, [self.title]+labels))
        if multi_header: table += self._to_latex_row(label=f'{self.title:>{n}}', data=multi_header)
        table += self._to_latex_row(label=f'{self.title:>{n}}', data=header)
        table += '\\midrule\n'
        for tag, row in zip(labels, body):
            table += self._to_latex_row(label=f'{tag:>{n}}', data=row)

        table += (
            '\\bottomrule\n'
            '\\end{tabular}\n'
            '\\end{table}\n'
            '% =====================================\n'
        )
        return table

    def to_markdown(self,
                    precision: int = 2,
                    width: ty.N[ty.U[int, ty.S[int]]] = None,
                    sort: ty.N[int] = None,
                    reverse: bool = False) -> str:
        """Create a Markdown table.

        :param precision: (int) Precision when rounding table `body`.
        :param width: (int) Row character width.
        :param sort: (None|int) Column index to sort by.
        :param reverse: (bool) If `True`, sort from worst to best.
        :return: (str) Markdown table represented as a string.
        """
        multi_header, header = self._get_header()
        labels, body, best_mask, nbest_mask = self.labels, self.body, self.best_mask, self.nbest_mask

        if sort is not None:
            labels, body, best_mask, nbest_mask = self._sort(sort, labels, body, best_mask, nbest_mask, reverse=reverse)

        body = np.vectorize(lambda i: f'{i:.{precision}f}')(body).astype('<U256')
        body[best_mask] = [f'<span style="color:green"><strong>{i}</strong></span>' for i in body[best_mask]]
        body[nbest_mask] = [f'<span style="color:blue;text-decoration: underline">{i}</span>' for i in body[nbest_mask]]

        multi_header, header, body = self._pad(header, multi_header, body, width)

        table = ''
        n = max(map(len, [self.title] + labels))
        if multi_header: table += self._to_md_row(label=f'{self.title:>{n}}', data=multi_header)
        table += self._to_md_row(label=f'{self.title:>{n}}', data=header)
        table += '|---|' + '---|'*(len(header)-1) + '\n'
        for tag, row in zip(labels, body):
            table += self._to_md_row(label=f'{tag:>{n}}', data=row)

        return table

