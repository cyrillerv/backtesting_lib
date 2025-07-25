# Métriques : sharpe, hit_ratio, drawdown, etc.
import numpy as np
import statsmodels.api as sm

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




def compute_factor_exposition(portfolio_returns, bench_df_input) :

    bench_returns = bench_df_input.pct_change()
    bench_returns.dropna(inplace=True)
    # bench_returns_reindex = bench_returns.reindex(portfolio_returns.index)

    X_factors = bench_returns.copy()
    y_strategy_returns = portfolio_returns.dropna(how='all').copy()
    # Assure-toi que les deux sont bien alignés dans le temps
    X_factors, y_strategy_returns = X_factors.align(y_strategy_returns, join="inner", axis=0)

    X_sm = sm.add_constant(X_factors)  # Ajoute une colonne "constante" pour alpha
    model_sm = sm.OLS(y_strategy_returns, X_sm).fit()

    return model_sm.summary()