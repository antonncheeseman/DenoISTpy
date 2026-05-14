from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(slots=True)
class DenoistResult:
    """Container returned by :func:`denoistpy.denoist`.

    Matrices are stored as genes x cells to mirror the R implementation.
    """

    adjusted_counts: sparse.spmatrix
    memberships: sparse.spmatrix | np.ndarray | None
    params: pd.DataFrame
    posterior: sparse.spmatrix | np.ndarray | None = None
    offsets: sparse.spmatrix | None = None
    gene_names: np.ndarray | None = None
    cell_names: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
