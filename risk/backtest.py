"""Backtesting de la VaR : exceptions, tests de Kupiec et Christoffersen, feux Bâle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def chi2_1_pvalue(stat: float) -> float:
    """p-value d'un χ² à 1 degré de liberté."""
    return 1.0 if stat <= 0 else math.erfc(math.sqrt(stat / 2))


def exceptions(pnl: pd.Series, var: pd.Series) -> pd.Series:
    """True les jours où la perte dépasse la VaR de la veille."""
    return (-pnl > var) & var.notna()


def kupiec_pof(n_exceptions: int, n_obs: int, confidence: float) -> tuple[float, float]:
    """Test de couverture inconditionnelle (proportion of failures)."""
    p = 1 - confidence
    x, n = n_exceptions, n_obs
    ph = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p)) + 2 * (_xlogy(n - x, 1 - ph) + _xlogy(x, ph))
    return lr, chi2_1_pvalue(lr)


def christoffersen_ind(hits) -> tuple[float, float, int]:
    """Test d'indépendance des exceptions (chaîne de Markov d'ordre 1).

    Renvoie (statistique, p-value, nombre d'exceptions consécutives).
    """
    h = np.asarray(hits, dtype=bool)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    p01 = n01 / max(1, n00 + n01)
    p11 = n11 / max(1, n10 + n11)
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    lr = -2 * (_xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)) + 2 * (
        _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    )
    return lr, chi2_1_pvalue(lr), n11


# Supplément au multiplicateur de 3 selon le nombre d'exceptions sur 250 jours (Bâle, VaR 99 %).
BASEL_PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_zone(n_exceptions: int) -> tuple[str, float]:
    """Zone des feux tricolores et supplément au multiplicateur (250 obs., 99 %)."""
    if n_exceptions <= 4:
        return "verte", 0.0
    if n_exceptions <= 9:
        return "jaune", BASEL_PLUS_FACTOR[n_exceptions]
    return "rouge", 1.0


FRTB_MAX_EXCEPTIONS = {0.99: 12, 0.975: 30}


@dataclass(frozen=True)
class BacktestResult:
    n_obs: int
    n_exceptions: int
    expected: float
    kupiec_lr: float
    kupiec_p: float
    ind_lr: float
    ind_p: float
    consecutive: int

    @property
    def conditional_p(self) -> float:
        """Test de couverture conditionnelle (χ² à 2 ddl)."""
        return math.exp(-(self.kupiec_lr + self.ind_lr) / 2)


def run_backtest(pnl: pd.Series, var: pd.Series, confidence: float) -> BacktestResult:
    mask = var.notna()
    hits = exceptions(pnl[mask], var[mask])
    n, x = int(hits.size), int(hits.sum())
    lr_uc, p_uc = kupiec_pof(x, n, confidence)
    lr_ind, p_ind, n11 = christoffersen_ind(hits.to_numpy())
    return BacktestResult(n, x, n * (1 - confidence), lr_uc, p_uc, lr_ind, p_ind, n11)
