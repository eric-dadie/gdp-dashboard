import math

import numpy as np
import pandas as pd
import pytest

from risk.backtest import basel_zone, christoffersen_ind, kupiec_pof, run_backtest
from risk.measures import expected_shortfall, historical_var, parametric_var, rolling_var
from risk.simulation import DESKS, simulate_pnl
from risk.stress import SCENARIOS, STRESS_LIMIT, breaking_multiplier, stress_pnl, total_stress_pnl


@pytest.fixture(scope="module")
def pnl():
    return simulate_pnl().sum(axis=1)


def test_simulation_shape_and_reproducibility():
    a, b = simulate_pnl(seed=3), simulate_pnl(seed=3)
    assert a.shape == (760, len(DESKS))
    pd.testing.assert_frame_equal(a, b)
    assert a.index.dayofweek.max() <= 4  # jours ouvrés uniquement


def test_historical_var_matches_quantile():
    x = np.arange(-50, 50, dtype=float)  # 100 P&L
    assert historical_var(x, 0.99) == pytest.approx(-np.quantile(x, 0.01))


def test_parametric_var_standard_normal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    assert parametric_var(x, 0.99) == pytest.approx(2.326, abs=0.02)


def test_expected_shortfall_averages_tail():
    x = np.array([-10, -8, -6, -4, -2] + [0] * 195, dtype=float)
    assert expected_shortfall(x, 0.975) == pytest.approx(30 / 5)  # ceil(200*2.5%) = 5


def test_es_at_least_var_on_fat_tails():
    rng = np.random.default_rng(1)
    x = rng.standard_t(3, 100_000)
    assert expected_shortfall(x, 0.975) > historical_var(x, 0.975)


def test_rolling_var_uses_only_past_data(pnl):
    v = rolling_var(pnl, "historical", 0.99, 250)
    assert v.iloc[:250].isna().all()
    assert v.iloc[300] == pytest.approx(historical_var(pnl.iloc[50:300], 0.99))


@pytest.mark.parametrize("model", ["historical", "parametric", "ewma"])
def test_rolling_var_positive(pnl, model):
    v = rolling_var(pnl, model, 0.99).dropna()
    assert (v > 0).all()


def test_kupiec_zero_when_rate_matches():
    lr, p = kupiec_pof(5, 500, 0.99)
    assert lr == pytest.approx(0, abs=1e-9)
    assert p == pytest.approx(1)


def test_kupiec_rejects_too_many_exceptions():
    _, p = kupiec_pof(12, 250, 0.99)
    assert p < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(250, dtype=bool)
    clustered[100:106] = True
    spread = np.zeros(250, dtype=bool)
    spread[[20, 60, 100, 140, 180, 220]] = True
    assert christoffersen_ind(clustered)[1] < 0.01
    assert christoffersen_ind(spread)[1] > 0.3


@pytest.mark.parametrize("n,zone,plus", [(0, "verte", 0), (4, "verte", 0), (5, "jaune", 0.40), (9, "jaune", 0.85), (10, "rouge", 1.0)])
def test_basel_traffic_light(n, zone, plus):
    assert basel_zone(n) == (zone, plus)


def test_run_backtest_counts(pnl):
    v = rolling_var(pnl, "historical", 0.99)
    r = run_backtest(pnl.iloc[-250:], v.iloc[-250:], 0.99)
    assert r.n_obs == 250
    assert r.n_exceptions == int((-pnl.iloc[-250:] > v.iloc[-250:]).sum())
    assert 0 <= r.conditional_p <= 1


def test_stress_zero_shock_is_zero():
    assert total_stress_pnl({}) == 0


def test_stress_rates_up_loses_on_long_duration():
    grid = stress_pnl({"ir": 100})
    assert grid.loc["Taux", "Taux"] == pytest.approx(-4.5)


def test_gamma_is_symmetric_loss():
    up, down = stress_pnl({"eq": 10}), stress_pnl({"eq": -10})
    assert up["Gamma actions"].sum() == pytest.approx(down["Gamma actions"].sum())
    assert up["Gamma actions"].sum() < 0


def test_breaking_multiplier_hits_limit():
    shocks = SCENARIOS[0].shocks
    k = breaking_multiplier(shocks)
    assert k is not None
    assert total_stress_pnl({f: v * k for f, v in shocks.items()}) == pytest.approx(-STRESS_LIMIT, abs=1e-6)
    assert breaking_multiplier({}) is None
    assert math.isfinite(k)
