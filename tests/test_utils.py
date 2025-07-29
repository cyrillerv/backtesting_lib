import sys
import os

# Ajoute le dossier parent (contenant 'backtesting') au chemin d'import
sys.path.append(os.path.abspath("..."))

import pandas as pd
import numpy as np
import pytest
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

    expected_pnl_ticker = pd.DataFrame({
        "AAPL": [np.nan, 20, -10, 20],
        "GOOG": [np.nan, -10, -5, 10]
    }, index=dates)
    expected_daily = pd.Series([0, 10.0, -15.0, 30.0], index=dates)
    expected_cumulative = expected_daily.cumsum()

    # Appel
    cumulative, daily, daily_pnl_per_ticker = compute_portfolio_pnl(volume, prices)

    # Vérifications
    pd.testing.assert_frame_equal(daily_pnl_per_ticker, expected_pnl_ticker, check_freq=False)
    pd.testing.assert_series_equal(daily, expected_daily)
    pd.testing.assert_series_equal(cumulative, expected_cumulative)

# TODO: mettre les df en brut au début du fichier pour ne les écrire qu'une seule fois.
def test_fct_avg_buying_price() :

    # Dates fictives pour l'index (tu peux adapter selon ton cas)
    dates = pd.date_range(start='2024-01-01', periods=14, freq='D')

    # volume_flows
    df_volume_flows = pd.DataFrame({
        'A': [0, 5, 2, -4, -3, 10, -15, 6, -2, -10, 11, -5, 9, -4],
        'B': [0, 5, 2, -4, -3, 10, -15, 6, -2, -10, 11, -5, 9, -4]
    }, index=dates)

    # volume en portefeuille
    df_volume_en_portefeuille = pd.DataFrame({
        'A': [0, 5, 7, 3, 0, 10, -5, 1, -1, -11, 0, -5, 4, 0],
        'B': [0, 5, 7, 3, 0, 10, -5, 1, -1, -11, 0, -5, 4, 0]
    }, index=dates)

    # stockPrices
    df_stock_price = pd.DataFrame({
        'A': [100, 101, 105, 104, 90, 92, 91, 92, 92, 101, 110, 112, 111, 109],
        'B': [100, 101, 105, 104, 90, 92, 91, 92, 92, 101, 110, 112, 111, 109]
    }, index=dates)


    # Création du DataFrame
    df_ABP_check = pd.DataFrame({
        'A': [
            0.000000, 101.000000, 102.142857, 102.142857, 0.000000,
            92.000000, 91.000000, 92.000000, 92.000000, 100.181818,
            0.000000, 112.000000, 111.000000, 0.000000
        ],
        'B': [
            0.000000, 101.000000, 102.142857, 102.142857, 0.000000,
            92.000000, 91.000000, 92.000000, 92.000000, 100.181818,
            0.000000, 112.000000, 111.000000, 0.000000
        ]
    }, index=pd.to_datetime([
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
        '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
        '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12',
        '2024-01-13', '2024-01-14'
    ]))

    df_ABP_check.index.name = 'date'
    df_ABP_check.columns.name = 'stock'


    res = calculate_avg_buying_price(df_volume_flows, df_volume_en_portefeuille, df_stock_price)

    # Harmonise les noms de colonnes/index pour une comparaison stricte
    res.columns.name = 'stock'
    res.index.name = 'date'

    # Vérifie l’égalité
    assert (res.round(5) == df_ABP_check.round(5)).all().all()


def test_fct_calc_profit_transaction() :

    # Dates fictives pour l'index (tu peux adapter selon ton cas)
    dates = pd.date_range(start='2024-01-01', periods=14, freq='D')

    # volume en portefeuille
    df_volume_en_portefeuille = pd.DataFrame({
        'A': [0, 5, 7, 3, 0, 10, -5, 1, -1, -11, 0, -5, 4, 0],
        'B': [0, 5, 7, 3, 0, 10, -5, 1, -1, -11, 0, -5, 4, 0]
    }, index=dates)

    # stockPrices
    df_stock_price = pd.DataFrame({
        'A': [100, 101, 105, 104, 90, 92, 91, 92, 92, 101, 110, 112, 111, 109],
        'B': [100, 101, 105, 104, 90, 92, 91, 92, 92, 101, 110, 112, 111, 109]
    }, index=dates)


    # Création du DataFrame
    df_ABP = pd.DataFrame({
        'A': [
            0.000000, 101.000000, 102.142857, 102.142857, 0.000000,
            92.000000, 91.000000, 92.000000, 92.000000, 100.181818,
            0.000000, 112.000000, 111.000000, 0.000000
        ],
        'B': [
            0.000000, 101.000000, 102.142857, 102.142857, 0.000000,
            92.000000, 91.000000, 92.000000, 92.000000, 100.181818,
            0.000000, 112.000000, 111.000000, 0.000000
        ]
    }, index=pd.to_datetime([
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
        '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
        '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12',
        '2024-01-13', '2024-01-14'
    ]))

    df_ABP.index.name = 'date'
    df_ABP.columns.name = 'stock'


    df_profit_long_check = pd.DataFrame({
        'A': [
            0, 0, 0, 0.018182, -0.118881,
            0,0,0,0,0,
            0, 0,0, -0.018018
        ],
        'B': [
            0, 0, 0, 0.018182, -0.118881,
            0,0,0,0,0,
            0, 0,0, -0.018018
        ],
    }, index=pd.to_datetime([
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
        '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
        '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12',
        '2024-01-13', '2024-01-14'
    ]))

    df_profit_short_check = pd.DataFrame({
        'A': [
            0, 0, 0, 0, 0,
            0,0,0,0,0,
            -0.098004, 0,0, 0
        ],
        'B': [
            0, 0, 0, 0, 0,
            0,0,0,0,0,
            -0.098004, 0,0, 0
        ],
    }, index=pd.to_datetime([
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
        '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
        '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12',
        '2024-01-13', '2024-01-14'
    ]))

    df_profit_long_check.index.name = 'date'
    df_profit_long_check.columns.name = 'stock'
    df_profit_short_check.index.name = 'date'
    df_profit_short_check.columns.name = 'stock'

    profit_long_positions, profit_short_positions = calculate_profit_transaction(df_volume_en_portefeuille, df_stock_price, df_ABP)

    profit_long_positions.fillna(0, inplace=True)
    profit_short_positions.fillna(0, inplace=True)

    # Harmonise les noms de colonnes/index pour une comparaison stricte
    profit_long_positions.columns.name = 'stock'
    profit_long_positions.index.name = 'date'
    profit_short_positions.columns.name = 'stock'
    profit_short_positions.index.name = 'date'

    # Vérifications
    pd.testing.assert_frame_equal(profit_long_positions, df_profit_long_check, check_freq=False)
    pd.testing.assert_frame_equal(profit_short_positions, df_profit_short_check, check_freq=False)




def format_df_orders(new_orders):
    new_orders = new_orders.copy()
    new_orders["Flows"] = new_orders["Volume"] * np.where(new_orders["Type"] == "Buy", 1, -1)
    new_orders.sort_values("Date", inplace=True)
    new_orders.reset_index(drop=True, inplace=True)
    return new_orders

@pytest.mark.parametrize("prices, new_orders_data, expected_result", [
    ## Buy order et TP triggered

    # 1. On achète 10, le TP se déclenche et on vend les 10
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.03},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 10, "Trigger": "TakeProfit"},
    ),

    # 2. On achète 10, mais il n'en reste que 5 car on en a vendu avant
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 3, "Trigger": "TakeProfit"},
    ),

    # 3. On achète 10 mais il n'en reste plus (on a tout vendu avant)
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        None,
    ),

    # 4. On achète 10 mais il y en a 13 dans notre portefeuille
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.05},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 10, "Trigger": "TakeProfit"},
    ),

    ## Buy order et SL triggered

    # 5. On achète 10, le TP se déclenche et on vend les 10
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.03, "TakeProfit": 0.03},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 10, "Trigger": "StopLoss"},
    ),

    # 6. On achète 10, mais il n'en reste que 5 car on en a vendu avant
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 3, "Trigger": "StopLoss"},
    ),

    # 7. On achète 10 mais il n'en reste plus (on a tout vendu avant)
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        None,
    ),

    # 8. On achète 10 mais il y en a 13 dans notre portefeuille
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.05},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Sell", "Volume": 10, "Trigger": "StopLoss"},
    ),

    ## Sell order et TP triggered

    # 9. On vend 10, le TP se déclenche et on rachète les 10
    (
        [100, 99, 98, 97, 96],  # prix baisse
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 10, "Trigger": "TakeProfit"},
    ),

    # 10.  On vend 10, mais il n'en reste que 5 car on en a racheté avant
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 3, "Trigger": "TakeProfit"},
    ),

    # 11. On vend 10 mais il n'en reste plus (on a tout racheté avant)
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        None,
    ),

    # 12. On vend 10 mais il y en a 13 dans notre portefeuille
    (
        [100, 99, 98, 97, 96],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.05},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 10, "Trigger": "TakeProfit"},
    ),

    ## Sell order et SL triggered

    # 13. On vend 10, le TP se déclenche et on rachète les 10
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 10, "Trigger": "StopLoss"},
    ),

    # 14. On vend 10, mais il n'en reste que 5 car on en a racheté avant
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 3, "Trigger": "StopLoss"},
    ),

    # 15. On vend 10 mais il n'en reste plus (on a tout racheté avant)
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 10, "Type": "Buy", "StopLoss": -0.02, "TakeProfit": 0.02},
        ],
        None,
    ),

    # 16. On vend 10 mais il y en a 13 dans notre portefeuille
    (
        [100, 101, 102, 103, 104],
        [
            {"Date": "2023-01-01", "Symbol": "AAPL", "Volume": 10, "Type": "Sell", "StopLoss": -0.03, "TakeProfit": 0.03},
            {"Date": "2023-01-03", "Symbol": "AAPL", "Volume": 7, "Type": "Sell", "StopLoss": -0.02, "TakeProfit": 0.05},
        ],
        {"Date": pd.Timestamp("2023-01-04"), "Type": "Buy", "Volume": 10, "Trigger": "StopLoss"},
    ),
])
def test_simulate_SL_TP_row(prices, new_orders_data, expected_result):
    # Préparer les données
    date_range = pd.date_range(start="2023-01-01", periods=len(prices), freq="D")
    df_stock_prices = pd.DataFrame({"AAPL": prices}, index=date_range)
    new_orders = pd.DataFrame(new_orders_data)
    new_orders["Date"] = pd.to_datetime(new_orders["Date"])
    orders_df_with_SL_TP = format_df_orders(new_orders)

    # Simulation
    row = new_orders.iloc[0]
    result = simulate_SL_TP_row(row, df_stock_prices, orders_df_with_SL_TP)
    print(result)

    # Vérification
    if expected_result is None:
        assert result is None
    else:
        for key, val in expected_result.items():
            assert result[key] == val
