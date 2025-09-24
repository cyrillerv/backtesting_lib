# Métriques : sharpe, hit_ratio, drawdown, etc.
import numpy as np
# import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

def compute_metrics(portfolio_returns, cumulative_pnl_portfolio, drawdown, df_volume_order, max_cash_needed, rf_annual, base):
    
    # Convertir en taux sans risque quotidien
    rf_daily = (1 + rf_annual)**(1/252) - 1
    average_return = portfolio_returns.mean()
    annualized_sharpe_ratio = (average_return - rf_daily) / portfolio_returns.std() * np.sqrt(base)
    annualized_sortino_ratio = (average_return - rf_daily) / portfolio_returns.clip(upper=0).std() * np.sqrt(base)
    annualized_calmar_ratio = (average_return - rf_daily) / drawdown.max()

    # Calcul calmar ratio
    # Calcul du rendement annualisé


    metrics = {}
    metrics["sharpe_ratio"] = round(float(annualized_sharpe_ratio), 2)
    metrics["sortino_ratio"] = round(float(annualized_sortino_ratio), 2)
    metrics["calmar_ratio"] = round(float(annualized_calmar_ratio), 2)
    metrics["max_cash_needed"] = round(float(max_cash_needed), 2)
    metrics["total_profit"] = round(float(cumulative_pnl_portfolio.iloc[-1]), 2)
    metrics["total_return"] = round(float(metrics["total_profit"] / metrics["max_cash_needed"]), 4)
    metrics["volatility"] = round(float(portfolio_returns.std(ddof=1) * np.sqrt(base)), 4) # ddof=1 signifie "degrés de liberté = 1" et sert à calculer l'écart-type corrigé (Bessel's correction), utilisé pour les échantillons au lieu de toute la population.
    metrics["max_drawdown"] = round(float(drawdown.max()), 4)
    metrics["average_transac_day"] = round(float(df_volume_order.notna().sum(axis=1).mean()), 2)
    metrics["max_nb_transac_day"] = int(df_volume_order.notna().sum(axis=1).max())
    metrics["nb_transacs_total"] = int(df_volume_order.notna().sum().sum())
    metrics["daily_portfolio_average_return"] = round(float(average_return), 4)
    
    return metrics




# def compute_factor_exposition(portfolio_returns, bench_df_input) :

#     bench_returns = bench_df_input.pct_change()
#     bench_returns.dropna(inplace=True)
#     # bench_returns_reindex = bench_returns.reindex(portfolio_returns.index)

#     X_factors = bench_returns.copy()
#     y_strategy_returns = portfolio_returns.dropna(how='all').copy()
#     # Assure-toi que les deux sont bien alignés dans le temps
#     X_factors, y_strategy_returns = X_factors.align(y_strategy_returns, join="inner", axis=0)

#     X_sm = sm.add_constant(X_factors)  # Ajoute une colonne "constante" pour alpha
#     model_sm = sm.OLS(y_strategy_returns, X_sm).fit()

#     return model_sm.summary()




def run_regression_factor_exposition(portfolio_returns, bench_df_input) :
    X_factors = bench_df_input.copy()
    y_strategy_returns = portfolio_returns.copy()

    y_strategy_returns.dropna(how='all', inplace=True)

    X, y = X_factors.align(y_strategy_returns, join="inner", axis=0)

    model = RidgeCV(alphas=[0.1, 0.5, 1.0, 5, 10.0, 20, 30, 50, 70, 100.0, 200, 300, 500, 600, 700, 800])
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    alpha = residuals.mean()

    r_squared = model.score(X, y)

    # Pour le graph
    coef_dict_regression = dict(zip(X.columns, model.coef_))
    # Comme metrics
    dic_metrics_regression = {"Ridge_alpha": model.alpha_, "alpha_generated": alpha, "R²": r_squared}

    return dic_metrics_regression, coef_dict_regression


def compute_metrics_per_ops(profit_long_positions, profit_short_positions) :

        all_perfs = (profit_long_positions.fillna(0) + profit_short_positions.fillna(0)).replace(0, np.nan)
    
        # TODO: gérer le cas où une opération a un profit de 0% pour le calcul du hit ratio, ce n'est pas pris en compe dans le dénominateur
        nb_long_winners = profit_long_positions.gt(0).sum().sum()
        nb_long_losers = profit_long_positions.lt(0).sum().sum()
        nb_short_winners = profit_short_positions.gt(0).sum().sum()
        nb_short_losers = profit_short_positions.lt(0).sum().sum()

        # Variables pour graph
        dic_winners_losers_long_short = {"Long - Gagnants": nb_long_winners, "Long - Perdants": nb_long_losers, "Short - Gagnants": nb_short_winners, "Short - Perdants":nb_short_losers}

        # Variables pour stats
        hit_ratio = (nb_long_winners + nb_short_winners) / (nb_long_winners + nb_long_losers + nb_short_winners + nb_short_losers)
        perfs_vals = all_perfs.values.ravel() # aplatir les data
        pos_perf = perfs_vals[perfs_vals > 0] 
        neg_perf = perfs_vals[perfs_vals < 0] 
        median_winners = np.nanmedian(pos_perf) if neg_perf.size else np.nan
        median_losers = np.nanmedian(neg_perf) if neg_perf.size else np.nan 
        average_proft_per_transaction = np.nanmean(all_perfs)
        dic_stats_operations = {"Hit_ratio": hit_ratio, "Winner_median": median_winners, "Loser_median": median_losers, "Avg_profit_per_ops": average_proft_per_transaction}

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