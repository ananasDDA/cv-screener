"""Helper package for `research.ipynb`.

The notebook stays thin: every measurement, every cache read and every chart lives here, the
same way `demo.ipynb` only calls into `cv_screener`. Raw results are cached as JSON under
`research/results/` so re-executing the notebook renders from disk and spends no API requests.
"""

from __future__ import annotations

__all__ = ["RESULTS_DIR"]

from .cache import RESULTS_DIR
