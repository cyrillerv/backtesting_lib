# Métriques : sharpe, hit_ratio, drawdown, etc.
import numpy as np

def compute_metrics(portfolio_returns, cumulative_pnl_portfolio, drawdown, df_volume_order, max_cash_needed, rf_annual, base):
    
    # Convertir en taux sans risque quotidien
    rf_daily = (1 + rf_annual)**(1/252) - 1
    # Sharpe Ratio annualisé 
    annualized_sharpe_ratio = (portfolio_returns.mean() - rf_daily) / portfolio_returns.std() * np.sqrt(base)
    annualized_sortino_ratio = (portfolio_returns.mean() - rf_daily) / portfolio_returns.clip(upper=0).std() * np.sqrt(base)
    annualized_calmar_ratio = (portfolio_returns.mean() - rf_daily) / drawdown.max()

    metrics = {}
    metrics["sharpe_ratio"] = annualized_sharpe_ratio
    metrics["sortino_ratio"] = annualized_sortino_ratio
    metrics["calmar_ratio"] = annualized_calmar_ratio
    metrics["max_cash_needed"] = max_cash_needed
    metrics["total_profit"] = cumulative_pnl_portfolio.iloc[-1]
    metrics["total_return"] = metrics["total_profit"] / metrics["max_cash_needed"]
    metrics["volatility"] = portfolio_returns.std(ddof=1) * np.sqrt(base) # ddof=1 signifie "degrés de liberté = 1" et sert à calculer l'écart-type corrigé (Bessel's correction), utilisé pour les échantillons au lieu de toute la population.
    metrics["max_drawdown"] = drawdown.max()
    metrics["average_transac_day"] = df_volume_order.notna().sum(axis=1).mean()
    metrics["max_nb_transac_day"] = df_volume_order.notna().sum(axis=1).max()
    metrics["nb_transacs_total"] = df_volume_order.notna().sum().sum()
    metrics["daily_portfolio_average_return"] = portfolio_returns.mean()
    
    return metrics

