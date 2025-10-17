# Métriques : sharpe, hit_ratio, drawdown, etc.
import numpy as np
# import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

def compute_metrics(portfolio_returns, cumulative_pnl_portfolio, drawdown, df_volume_order, max_cash_needed, rf_annual, base):
    
    # Count the number of years
    nb_years = (portfolio_returns.index.max() - portfolio_returns.index.min()) / pd.Timedelta(days=365.25)

    # Calculate annualized return
    total_profit = cumulative_pnl_portfolio.iloc[-1]
    total_return = total_profit / max_cash_needed
    annualized_return = total_return ** (1 / nb_years) - 1
    annualized_stdev = portfolio_returns.std() * np.sqrt(base)
    annualized_downside_stdev = portfolio_returns.clip(upper=0).std() * np.sqrt(base)

    average_return = portfolio_returns.mean()

    excess_return = annualized_return - rf_annual
    annualized_sharpe_ratio = (excess_return / annualized_stdev
                            if annualized_stdev != 0 else np.nan)

    # downside_std = portfolio_returns.clip(upper=0).std()
    annualized_sortino_ratio = (excess_return / annualized_downside_stdev
                                if annualized_downside_stdev != 0 else np.nan)

    calmar_ratio = (annualized_return / drawdown.max()
                            if drawdown.max() != 0 else np.nan)


    def format_metric(value, decimals):
        """Format a metric value with appropriate precision."""
        return int(value) if decimals == 0 else round(float(value), decimals)

    metrics = {
        "sharpe_ratio": format_metric(annualized_sharpe_ratio, 2),
        "sortino_ratio": format_metric(annualized_sortino_ratio, 2),
        "calmar_ratio": format_metric(calmar_ratio, 2),
        "max_cash_needed": format_metric(max_cash_needed, 2),
        "total_profit": format_metric(total_profit, 2),
        "total_return": format_metric(total_return, 4),
        "annualized_return": format_metric(annualized_return, 4),
        "volatility": format_metric(portfolio_returns.std(ddof=1) * np.sqrt(base), 4),
        "max_drawdown": format_metric(drawdown.max(), 4),
        "average_transac_day": format_metric(df_volume_order.notna().sum(axis=1).mean(), 2),
        "max_nb_transac_day": format_metric(df_volume_order.notna().sum(axis=1).max(), 0),
        "nb_transacs_total": format_metric(df_volume_order.notna().sum().sum(), 0),
        "daily_portfolio_average_return": format_metric(average_return, 4)
    }
    
    return metrics


from sklearn.linear_model import LinearRegression

def run_ols_factor_regression(strategy_returns, factor_returns):
    """
    Perform OLS factor regression analysis.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy returns time series
    factor_returns : pd.DataFrame
        Factor returns (already in returns, not prices)
        
    Returns
    -------
    dict
        Regression metrics (alpha, R², residuals std, etc.)
    dict
        Factor exposures (betas)
    pd.Series
        Residuals
    pd.Series
        Predicted returns
    """
    # Align data on common dates
    X, y = factor_returns.align(strategy_returns, join="inner", axis=0)
    
    if len(X) == 0:
        raise ValueError("No overlapping dates between strategy and factor returns")
    
    # OLS regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions and residuals
    y_pred = pd.Series(model.predict(X), index=y.index)
    residuals = y - y_pred
    
    # Calculate metrics
    alpha = model.intercept_
    r_squared = model.score(X, y)
    residuals_std = residuals.std()
    
    metrics = {
        "alpha": round(float(alpha), 6),
        "alpha_annualized": round(float(alpha * 252), 6),  # Assuming daily returns
        "r_squared": round(float(r_squared), 4),
        "residuals_std": round(float(residuals_std), 6),
        "n_observations": len(X)
    }
    
    factor_exposures = {
        factor: round(float(beta), 4) 
        for factor, beta in zip(X.columns, model.coef_)
    }
    
    return metrics, factor_exposures, residuals, y_pred




def compute_metrics_per_ops(profit_long_positions, profit_short_positions, df_volume_portfolio) :

        all_perfs = (profit_long_positions.fillna(0) + profit_short_positions.fillna(0)).replace(0, np.nan)
    
        # TODO: gérer le cas où une opération a un profit de 0% pour le calcul du hit ratio, ce n'est pas pris en compe dans le dénominateur
        nb_long_winners = profit_long_positions.gt(0).sum().sum()
        nb_long_losers = profit_long_positions.lt(0).sum().sum()
        nb_short_winners = profit_short_positions.gt(0).sum().sum()
        nb_short_losers = profit_short_positions.lt(0).sum().sum()

        # Variables pour graph
        dic_winners_losers_long_short = {"Long - Gagnants": nb_long_winners, "Long - Perdants": nb_long_losers, "Short - Gagnants": nb_short_winners, "Short - Perdants":nb_short_losers}

        # Fonction pour regrouper les valeurs séparées par des NaN
        def group_by_nan(series):
            """
            Fonction pour regrouper les valeurs séparées par des NaN et de compter la taille du groupe.
            """
            group_id = series.isna().cumsum()
            filled_series = series.fillna(-1)  # Remplacer temporairement les NaN
            groups = filled_series[filled_series != -1].groupby(group_id).count()
            return groups

        # Calcul de la médiane des holding period par ops
        df_len_ops = df_volume_portfolio.replace(0, np.nan).apply(group_by_nan)
        valid_holding_period = perfs_vals[~np.isnan(perfs_vals)]
        median_holding_period = np.nan if valid_holding_period.size == 0 else np.nanmedian(df_len_ops.values.flatten())

        # Variables pour stats
        total_trades = nb_long_winners + nb_long_losers + nb_short_winners + nb_short_losers
        hit_ratio = ((nb_long_winners + nb_short_winners) / total_trades
                    if total_trades > 0 else np.nan)
        perfs_vals = all_perfs.values.ravel() # aplatir les data
        pos_perf = perfs_vals[perfs_vals > 0] 
        neg_perf = perfs_vals[perfs_vals < 0] 
        median_winners = np.nanmedian(pos_perf) if pos_perf.size else np.nan
        median_losers = np.nanmedian(neg_perf) if neg_perf.size else np.nan 
        valid_perfs = perfs_vals[~np.isnan(perfs_vals)]
        average_proft_per_transaction = np.nan if valid_perfs.size == 0 else np.nanmean(valid_perfs)
        dic_stats_operations = {"Hit_ratio": hit_ratio, "Winner_median": median_winners, "Loser_median": median_losers, "Avg_profit_per_ops": average_proft_per_transaction, "Median_holding_period":median_holding_period}

        return dic_winners_losers_long_short, dic_stats_operations



import pandas as pd
# TODO: ne calculer que les metrics ici et mettre les calculs de df dans le file positions.py
def metrics_tickers(df_valeur_order, daily_pnl_per_ticker) :    
    # TODO: on affiche toutes les opérations même celles qui ne sont pas clôturées in fine => réfléchir si on garde ça comme ça ou si on n'affiche que les opérations clôturées.
    ## Statistiques par ticker
    nb_transactions_par_ticker = df_valeur_order.notna().sum()

    # On calcule la qty en euros allouée au total pour chaque ticker et on enlève ceux pour qui on en a 0 (les tickers dont les positions n'ont pas été clôturée)
    total_amount_allocated_per_ticker = df_valeur_order.abs().sum()
    # total_amount_allocated_per_ticker = total_amount_allocated_per_ticker[total_amount_allocated_per_ticker.ne(0)]

    total_return_per_ticker = daily_pnl_per_ticker.sum() / total_amount_allocated_per_ticker
    df_metrics_per_ticker = pd.concat([nb_transactions_par_ticker.rename("Nb transacs"), total_amount_allocated_per_ticker.rename("Volume $"), total_return_per_ticker.rename("Total_return")], axis=1).dropna()

    best_performer_ticker = df_metrics_per_ticker.idxmax()["Total_return"]
    best_performer_value = df_metrics_per_ticker.max()["Total_return"]

    max_invested_ticker = df_metrics_per_ticker.abs().idxmax()["Volume $"]
    max_invested_value = df_metrics_per_ticker.abs().max()["Volume $"]

    max_operations_ticker  = df_metrics_per_ticker.idxmax()["Nb transacs"]
    max_operations_value  = df_metrics_per_ticker.max()["Nb transacs"]

    # C'est le df qui récap les tickers avec les plus de ....
    df_metrics_max_tickers = pd.DataFrame({"Max_invested":{"Ticker": max_invested_ticker, "value": max_invested_value}, "Best_perfomer": {"Ticker": best_performer_ticker, "value": best_performer_value}, "Max_operations": {"Ticker": max_operations_ticker, "value": max_operations_value}}).T

    return df_metrics_per_ticker, df_metrics_max_tickers
    # Pour graphs
    self.df_metrics_per_ticker = df_metrics_per_ticker
    # Pour print
    self.df_metrics_max_tickers = df_metrics_max_tickers