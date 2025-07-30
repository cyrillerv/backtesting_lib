# Fonctions auxiliaires (ex: to_cumprod, normalisation, etc.)
import pandas as pd
import numpy as np

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
    # TODO: réfléchir si on ne peut pas renvoyer que deux variables au lieu de 3
    daily_pnl_per_ticker = (portfolio_volume.shift(1) * stock_prices.diff())
    daily_pnl = daily_pnl_per_ticker.sum(axis=1)
    cumulative_pnl = daily_pnl.cumsum()

    return cumulative_pnl, daily_pnl, daily_pnl_per_ticker


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


# TODO: changer le nom et enlever buying et mettre opening price instead

def calculate_avg_buying_price(df_volume_flows, df_volume_en_portefeuille, df_stock_price) :
    """
    Crée le df qui affiche l'average buying (or selling) price pour chaque date où une opération est effectuée
    """

    # 1. On calcule les ABP qui ne nécessitent pas de passer par une boucle 
    condition_open_position = (((df_volume_en_portefeuille.shift(1) == 0) | (pd.isna(df_volume_en_portefeuille.shift(1)))) & (df_volume_en_portefeuille != 0)) | (df_volume_en_portefeuille.shift(1) * df_volume_en_portefeuille < 0)
    # ABP quand on ouvre une position
    ABP_open_position = df_stock_price[condition_open_position].copy()


    # Lorsqu'on renforce notre position après en avoir ouvert une juste avant
    condition_renfo_after_open = ~condition_open_position & condition_open_position.shift(1) & (df_volume_flows * df_volume_en_portefeuille.shift(1) > 0)
    ABP_renfo_after_open = ((df_volume_flows * df_stock_price + df_volume_en_portefeuille.shift(1) * df_stock_price.shift(1)) / df_volume_en_portefeuille)[condition_renfo_after_open].copy()


    # 2. On filtre les lignes qui devront passer dans la boucle
    # Condition qui filtre pour ne garder que les opérations qui réduisent la position en cours 
    condition_reduction_pos = (df_volume_flows * df_volume_en_portefeuille.shift(1) < 0) & (df_volume_en_portefeuille * df_volume_en_portefeuille.shift(1) > 0)

    # On ne garde que les lignes pour lesquelles on renfo la position mais que l'opération précédente n'était pas un open position.
    # Le replace nan qu'on fait n'a pas d'importance car ça sera pour la première opération et elle n'est forcément pas concernée car on ne garde que les renfo pour lesquels pas d'open position juste avant = > minimum 2 opérations faites au préalable.
    condition_renfo_last_not_open = (~condition_open_position.shift(1).replace(np.nan, False)) & (df_volume_en_portefeuille * df_volume_en_portefeuille.shift(1) > 0) & (df_volume_flows * df_volume_en_portefeuille.shift(1) > 0)


    ABP_before_loop = ABP_open_position.fillna(0) + ABP_renfo_after_open.fillna(0)

    dic_ABP = {}

    for idx in df_volume_flows.index:
        df_volume_flows_filtered = df_volume_flows.loc[idx].dropna().copy()
        for stock in df_volume_flows_filtered.index:

            key = (idx, stock)

            # Ancienne valeur de ABP
            stock_keys = [key[0] for key in dic_ABP if key[1] == stock]
            prev_key = (max(stock_keys) if stock_keys else None, stock)

            abp_prev = dic_ABP.get(prev_key, 0)

            # Si on est sur une réduction de position, alors on ne change pas l'ABP => changer si on veut faire du FIFO.
            if condition_reduction_pos.loc[idx, stock]:
                dic_ABP[key] = abp_prev

            # Si on renforce notre position et la transaction précédente n'était pas un open (donc un autre renfo), alors on fait ce calcul.
            elif condition_renfo_last_not_open.loc[idx, stock]:
                numerator = (
                    df_volume_flows.loc[idx, stock] * df_stock_price.loc[idx, stock]
                    + df_volume_en_portefeuille.loc[prev_key[0], stock] * abp_prev
                )
                denominator = df_volume_en_portefeuille.loc[idx, stock]
                dic_ABP[key] = numerator / denominator if denominator != 0 else 0

            # Si ce n'est aucun des ces deux cas là, alors on l'a forcément déjà calculé hors boucle.
            else:
                dic_ABP[key] = ABP_before_loop.loc[idx, stock]


    # 1. Convertir le dictionnaire en DataFrame
    df_ABP = pd.DataFrame.from_dict(dic_ABP, orient='index', columns=['ABP'])

    # 2. Transformer l’index tuple en MultiIndex propre
    df_ABP.index = pd.MultiIndex.from_tuples(df_ABP.index, names=["date", "stock"])

    # 3. (Optionnel) Remettre en forme : un index = date, colonnes = stock
    df_ABP = df_ABP.unstack("stock")  # chaque colonne devient ABP d’un stock

    # 4. (Optionnel) Renommer colonnes
    df_ABP.columns = df_ABP.columns.droplevel(0)  # retire le niveau "ABP" si inutile

    # On vérifie qu'il n'y ait aucun buying price négatif
    assert (df_ABP < 0).sum().sum() == 0, f"Problème dans les calculs de l'average buying price: Il y a des prix négatifs."
    
    return df_ABP

def calculate_profit_transaction(df_volume_en_portefeuille, df_stock_price, df_ABP) : 
    """
    1. On met l'ABP au closing dates, que ce soit un total close ou un partial.
    2. On calcule la variation entre l'ABP et le closing price.
    3. On multiplie la variation par la position closed (1 pour long et -1 pour short)
    """
    ## 1.
    # Indicateur pour les close
    # Si la valeur est égale à 0 et que la valeur précédente n'était pas 0, alors onn a une clôture totale
    indicateur_total_close = (df_volume_en_portefeuille.eq(0) & df_volume_en_portefeuille.shift(1).ne(0) & df_volume_en_portefeuille.shift(1).notna())
    # Si la valeur est différente de 0 et est inférieure à la valeur précédente et qu'on garde la même position, alors on a une clôture partielle
    # Si la position actuelle est non nulle ET que le volume en valeur abs est inférieur au volume de la veille en abs ET que le volume d'ajourd'hui est du même signe que hier
    indicateur_partial_close = df_volume_en_portefeuille.ne(0) & df_volume_en_portefeuille.abs().lt(df_volume_en_portefeuille.shift(1).abs()) & (df_volume_en_portefeuille * df_volume_en_portefeuille.shift(1) > 0)

    indicateur_close = indicateur_total_close + indicateur_partial_close
    # Pour chaque close, on met l'ABP de l'open
    # On fait shift(1) car l'ABP est calculée après le close pour la ligne en question
    df_ABP_closing_dates = df_ABP.reindex(df_volume_en_portefeuille.index).replace(0, np.nan).ffill().shift(1)[indicateur_close].copy()

    ## 2.
    var_on_closing_date = df_stock_price[indicateur_close] / df_ABP_closing_dates - 1

    ## 3.
    # On est obligé de faire deux df séparés (un close long et un close short) si on close un short et un long en même temps, on ne peut pas aggréger car on ne sell pas nécessairement ce qu'on a acheter (stop loss et take profit).
    # Mettre des 1 partout où on close un long, et des nan autrement.
    data = np.where(df_volume_en_portefeuille.shift(1)[indicateur_close] > 0, 1, np.nan)
    df_indic_long_close = pd.DataFrame(data=data, index=df_volume_en_portefeuille.index, columns=df_volume_en_portefeuille.columns)
    # Mettre des -1 partout où on close un short, et des nan autrement.
    data = np.where(df_volume_en_portefeuille.shift(1)[indicateur_close] < 0, -1, np.nan)
    df_indic_short_close = pd.DataFrame(data=data, index=df_volume_en_portefeuille.index, columns=df_volume_en_portefeuille.columns)

    profit_long_positions = var_on_closing_date * df_indic_long_close
    profit_short_positions = var_on_closing_date * df_indic_short_close

    return profit_long_positions, profit_short_positions




def simulate_SL_TP_row(row, df_stock_prices, orders_df_with_SL_TP):
    """
    Simule un stop loss (SL) ou un take profit (TP) pour une ligne d'ordre.
    Retourne une ligne d'ordre inverse si SL ou TP est déclenché.
    """
    # TODO: voir combien le déclenchement du SL ou du TP a permi de gagner
    # Pour faire le FIFO, on prend à partir de la date de transaction jusq'uà ce qu'on ait 0 stocks en portefeuille. (actuel)
    # Pour faire le LIFO, au lieu de regarder jusqu'à quand on a plus de stocks en portefeuille, on va regarder jusqu'à quand on a nos stocks
    # ex: si on achète 5 mais qu'on en avait déjà 10, on regarder jusq'uà ce qu'on retombe sur 10 (et donc que nos 5 soient écoulés)
    # Il suffit de changer l'endate

    # On isole l'évolution du nombre d'actions en portefeuille au cours du temps pour ce ticker
    volume_en_portfolio = orders_df_with_SL_TP[orders_df_with_SL_TP['Symbol'] == row['Symbol']].sort_values("Date").set_index("Date")["Flows"].cumsum()
    start_date = row['Date']
    volume_en_portfolio = volume_en_portfolio.loc[start_date:].copy()
    # Le premier chiffre négatif ou nul signifie qu'on a clôturer la position à cette date
    end_date = (volume_en_portfolio * volume_en_portfolio.shift(1))[(volume_en_portfolio * volume_en_portfolio.shift(1)) <= 0].index.min()

    symbol = row['Symbol']
    
    if end_date :
        partial_prices = df_stock_prices.loc[start_date:end_date, symbol].copy()
    else : 
        partial_prices = df_stock_prices.loc[start_date:, symbol].copy()

    if partial_prices.empty:
        return None  # Rien à faire s'il n'y a pas de données

    entry_price = partial_prices.iloc[0]

    if row["Type"] == "Buy" :
        TP_price = entry_price * (1 + row['TakeProfit'])
        SL_price = entry_price * (1 + row['StopLoss'])

        TP_triggered = partial_prices[partial_prices >= TP_price]
        SL_triggered = partial_prices[partial_prices <= SL_price]
    else :
        TP_price = entry_price * (1 - row['TakeProfit'])
        SL_price = entry_price * (1 - row['StopLoss'])

        TP_triggered = partial_prices[partial_prices <= TP_price]
        SL_triggered = partial_prices[partial_prices >= SL_price]        

    # Init
    date_close = pd.Timestamp.max
    SL_trigger, TP_trigger = False, False

    if not TP_triggered.empty:
        date_close = TP_triggered.index[0]
        TP_trigger = True

    if not SL_triggered.empty:
        date_SL = SL_triggered.index[0]
        if date_SL < date_close:
            date_close = date_SL
            TP_trigger = False
            SL_trigger = True

    # On récupère le dernier volume qu'on avait en portefeuille avant la clôture
    volume_on_peut_close = abs(volume_en_portfolio[volume_en_portfolio.index <= date_close].iloc[-1])
    if SL_trigger or TP_trigger:
        return {
            "Date": date_close,
            "Symbol": symbol,
            "Volume": min(row['Volume'], volume_on_peut_close), # On prend le min entre le volume en portfolio et le volume sur lequel on a mis un trigger
            "Type": "Sell" if row['Type'] == "Buy" else "Buy",
            "StopLoss": np.nan,
            "TakeProfit": np.nan,
            "Trigger": "StopLoss" if SL_trigger else "TakeProfit"
        }

    return None