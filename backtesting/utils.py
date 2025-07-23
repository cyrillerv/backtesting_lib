# Fonctions auxiliaires (ex: to_cumprod, normalisation, etc.)



def calc_other_df(df_volume_portfolio, df_stock_prices_reindexed) :
    pnl_daily_ticker = df_volume_portfolio.shift(1) * df_stock_prices_reindexed.diff()
    daily_pnl_portfolio = pnl_daily_ticker.sum(axis=1) # pnl pour chaque date et chaque ticker
    cumulative_pnl_portfolio = daily_pnl_portfolio.cumsum()
    return cumulative_pnl_portfolio, daily_pnl_portfolio


def calc_returns(daily_pnl_portfolio, df_valeur_portfolio) :
    # Utiliser daily_pnl_portfolio en numérateur (plutôt que df_valeur_portfolio.diff()) permet d'effacer le cas où l'on rajoute des cash flows
    portfolio_returns = daily_pnl_portfolio / df_valeur_portfolio.shift(1).sum(axis=1)
    return portfolio_returns


def calculate_drawdown(cumulative_pnl_portfolio, max_cash_needed):
    """
    Calcule le drawdown en pourcentage à partir du PnL cumulé.

    Parameters:
    cumulative_pnl_portfolio (pd.Series): Série temporelle du PnL cumulé.

    Returns:
    pd.Series: Série temporelle du drawdown en pourcentage.
    """
    pnl_with_init_cash = max_cash_needed + cumulative_pnl_portfolio
    rolling_max = pnl_with_init_cash.cummax()
    drawdown = (pnl_with_init_cash - rolling_max) / rolling_max
    drawdown = -drawdown
    return drawdown.fillna(0)  # Si le rolling_max est 0, on évite les NaN



def calc_top_contributors(daily_pnl_portfolio, pnl_daily_ticker) : 
    """
    On regarde la pire perf et on regarde quels tickers y ont le plus participés. Même chose pour la meilleure perf.
    Objectif de faire un bar chart.
    """
    # On récupère la date de la worst perf
    date_worst_perf = daily_pnl_portfolio.idxmin()
    # C'est la contribution de chaque tickers au MDD
    contrib_ticker_to_worst_perf = pnl_daily_ticker.loc[pnl_daily_ticker.index == date_worst_perf].min() / pnl_daily_ticker.loc[pnl_daily_ticker.index == date_worst_perf].min().min()
    # On ne garde que ceux qui y ont le plus contribués
    top_contributors_worst_perf = contrib_ticker_to_worst_perf.loc[contrib_ticker_to_worst_perf > 0.2]


    # On récupère la date de la worst perf
    date_best_perf = daily_pnl_portfolio.idxmax()
    # C'est la contribution de chaque tickers au MDD
    contrib_ticker_to_best_perf = pnl_daily_ticker.loc[pnl_daily_ticker.index == date_best_perf].max() / pnl_daily_ticker.loc[pnl_daily_ticker.index == date_best_perf].max().max()
    # On ne garde que ceux qui y ont le plus contribués
    top_contributors_best_perf = contrib_ticker_to_best_perf.loc[contrib_ticker_to_best_perf > 0.2]

    return top_contributors_worst_perf, top_contributors_best_perf
