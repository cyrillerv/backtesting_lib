import sys
import os

# Ajoute le dossier parent (contenant 'backtesting') au chemin d'import
sys.path.append(os.path.abspath("..."))

import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from backtesting.utils import * 

def test_compute_top_contributors():
    # Exemple de données factices
    dates = pd.date_range("2024-01-01", periods=3)
    tickers = ['AAPL', 'GOOG', 'TSLA']

    pnl_by_ticker = pd.DataFrame([
        [10, -5, 2],
        [-50, -20, -10],  # Worst day (total: -80)
        [40, 30, 10]      # Best day (total: 80)
    ], index=dates, columns=tickers)

    daily_pnl = pnl_by_ticker.sum(axis=1)

    worst, best = compute_top_contributors(daily_pnl, pnl_by_ticker)

    # On s'attend à ce que le plus gros contributeur du worst day soit AAPL (-50 sur -80)
    assert 'AAPL' in worst.index
    assert worst['AAPL'] == 1.0  # Car normalisé par le min

    # On s'attend à ce que le plus gros contributeur du best day soit AAPL (40 sur 80)
    assert 'AAPL' in best.index
    assert best['AAPL'] == 1.0

    # Vérifie que la fonction retourne des Series
    assert isinstance(worst, pd.Series)
    assert isinstance(best, pd.Series)

    # Vérifie que les index sont bien des sous-ensembles des tickers
    assert set(worst.index).issubset(tickers)
    assert set(best.index).issubset(tickers)


def test_compute_drawdown_negative_percentage():
    # Série PnL simple avec un drawdown clair
    pnl = pd.Series([0, 100, 200, 100, 250], index=pd.date_range("2024-01-01", periods=5))
    initial_cash = 1000.0

    expected_equity = pnl + initial_cash
    rolling_max = expected_equity.cummax()
    expected_drawdown = -(expected_equity - rolling_max) / rolling_max
    expected_drawdown = expected_drawdown.fillna(0)

    actual_drawdown = compute_drawdown(pnl, initial_cash)

    # Vérifie que c'est bien négatif ou nul partout
    assert (actual_drawdown >= 0).all()

    # Vérifie l'égalité exacte
    pd.testing.assert_series_equal(actual_drawdown, expected_drawdown)


def test_compute_portfolio_returns_basic():
    dates = pd.date_range("2024-01-01", periods=4)
    
    # Exemple simple : 2 actifs avec valeurs claires
    portfolio_value = pd.DataFrame({
        "AAPL": [100, 110, 120, 130],
        "GOOG": [200, 190, 180, 170]
    }, index=dates)

    # Daily PnL supposé
    daily_pnl = pd.Series([0, 15, 20, 25], index=dates)

    # Calcul manuel des valeurs de référence
    lagged_total = portfolio_value.shift(1).sum(axis=1)
    expected_returns = daily_pnl / lagged_total

    # Appel fonction
    result = compute_portfolio_returns(daily_pnl, portfolio_value)

    # Vérification
    pd.testing.assert_series_equal(result, expected_returns)


def test_compute_portfolio_pnl_basic():
    dates = pd.date_range("2024-01-01", periods=4)

    # Portfolio avec 2 actions
    volume = pd.DataFrame({
        "AAPL": [10, 10, 10, 10],
        "GOOG": [5, 5, 5, 5]
    }, index=dates)

    prices = pd.DataFrame({
        "AAPL": [100, 102, 101, 103],
        "GOOG": [200, 198, 197, 199]
    }, index=dates)

    # Calcul attendu :
    # day 1: NaN
    # day 2: (10 * (102-100)) + (5 * (198-200)) = 20 - 10 = 10
    # day 3: (10 * (101-102)) + (5 * (197-198)) = -10 - 5 = -15
    # day 4: (10 * (103-101)) + (5 * (199-197)) = 20 + 10 = 30

    expected_daily = pd.Series([0, 10.0, -15.0, 30.0], index=dates)
    expected_cumulative = expected_daily.cumsum()

    # Appel
    cumulative, daily = compute_portfolio_pnl(volume, prices)

    # Vérifications
    pd.testing.assert_series_equal(daily, expected_daily)
    pd.testing.assert_series_equal(cumulative, expected_cumulative)