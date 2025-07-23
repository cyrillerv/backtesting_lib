# Fonctions auxiliaires (ex: to_cumprod, normalisation, etc.)
import pandas as pd

def compute_portfolio_pnl(
    portfolio_volume: pd.DataFrame,
    stock_prices: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Compute daily and cumulative PnL of a portfolio.

    Parameters ----------
    portfolio_volume : pd.DataFrame
        Portfolio positions (dates × tickers).
    stock_prices : pd.DataFrame
        Corresponding stock prices (same shape as portfolio_volume).

    Returns -------
    cumulative_pnl : pd.Series
        Cumulative profit and loss.
    daily_pnl : pd.Series
        Daily profit and loss.
    """

    daily_pnl = (portfolio_volume.shift(1) * stock_prices.diff()).sum(axis=1)
    cumulative_pnl = daily_pnl.cumsum()

    return cumulative_pnl, daily_pnl


def compute_portfolio_returns(
    daily_pnl: pd.Series,
    portfolio_value: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio daily returns using daily PnL and lagged portfolio value.

    Parameters ----------
    daily_pnl : pd.Series
        Daily profit and loss of the portfolio (indexed by date).
    portfolio_value : pd.DataFrame
        Portfolio value per asset (dates × tickers).

    Returns -------
    pd.Series
        Daily portfolio returns.
    """

    lagged_total_value = portfolio_value.shift(1).sum(axis=1)
    portfolio_returns = daily_pnl / lagged_total_value

    return portfolio_returns


def compute_drawdown(
    cumulative_pnl: pd.Series,
    initial_cash: float
) -> pd.Series:
    """
    Compute the portfolio drawdown as a percentage.

    Parameters ----------
    cumulative_pnl : pd.Series
        Cumulative profit and loss over time (indexed by date).
    initial_cash : float
        Initial capital committed to the strategy.

    Returns -------
    pd.Series
        Drawdown series as negative percentages.
    """
    equity_curve = cumulative_pnl + initial_cash
    rolling_max = equity_curve.cummax()
    drawdown = -(equity_curve - rolling_max) / rolling_max

    return drawdown.fillna(0)


def compute_top_contributors(
    daily_pnl: pd.Series,
    pnl_by_ticker: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Identify top-contributing tickers to the best and worst daily PnL.

    Parameters ----------
    daily_pnl : pd.Series
        Daily total PnL of the portfolio.
    pnl_by_ticker : pd.DataFrame
        Daily PnL per ticker (same index as daily_pnl).

    Returns -------
    tuple[pd.Series, pd.Series]
        Top contributors to the worst and best daily performances.
    """
    # Worst-performing day
    date_worst = daily_pnl.idxmin()
    daily_contrib_worst = pnl_by_ticker.loc[date_worst]
    contrib_ratio_worst = daily_contrib_worst / daily_contrib_worst.min()
    top_worst_contributors = contrib_ratio_worst[contrib_ratio_worst > 0.2]

    # Best-performing day
    date_best = daily_pnl.idxmax()
    daily_contrib_best = pnl_by_ticker.loc[date_best]
    contrib_ratio_best = daily_contrib_best / daily_contrib_best.max()
    top_best_contributors = contrib_ratio_best[contrib_ratio_best > 0.2]

    return top_worst_contributors, top_best_contributors