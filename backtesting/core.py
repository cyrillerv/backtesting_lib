import pandas as pd
from .positions import PortfolioBuilder
from .utils import (
    compute_portfolio_pnl,
    compute_portfolio_returns,
    compute_drawdown,
    calculate_avg_buying_price,
    calculate_profit_transaction
)
from .metrics import *
from .plotting import *

from .data_validator import DataValidator

# TODO: ajouter arg FIFO ou LIFO pour le SL & TP
# TODO: ajouter possiblité de choisir régression entrre OLS et Ridge (si OLS, faire une PCA dans DataValidator)
# TODO: changer le nom de la colonne 'Symbol' pour mettre 'Ticker qui est plus explicite.
# TODO: changer borrow_rate par qc de plus explicite
class BacktestEngine:
    def __init__(self, 
                 orders_df_input: pd.DataFrame, 
                 prices_df_input: pd.DataFrame, 
                 bench_df_input: pd.DataFrame = pd.DataFrame(),
                 stop_loss: float = np.nan,
                 take_profit: float = np.nan,
                 close_all: bool = True, 
                 transac_fees: float = 0.001, 
                 borrow_rate: float = 0.02, 
                 borrowing_cash_fees: float = 0.01,
                 loan_cash_fees: float = 0.0,
                 maintenance_margin_pct: float = 1.3,
                 annual_discount_rate: float = 0.03,
                 base: int = 252):
        """
        orders_df_input (pd.DataFrame): 
            DataFrame containing the trading orders executed by the strategy.
            Expected columns:
                - 'Date' (datetime): date of the order
                - 'Symbol' (str): asset ticker
                - 'Volume' (int > 0): quantity of the asset traded
                - 'Type' (str): 'Buy' or 'Sell'

        prices_df_input (pd.DataFrame): 
            DataFrame containing historical prices of the traded assets.
            - Index: datetime (observation dates)
            - Columns: asset tickers (must match those in `orders_df_input`)
            - Values: float (asset price at the given date)

        bench_df_input (pd.DataFrame, optional): 
            DataFrame containing prices of a benchmark (e.g., market index or reference asset).
            - Default: empty DataFrame
            - Index: datetime
            - Columns: benchmark tickers
            - Values: float (benchmark price at the given date)

        stop_loss (float < 0, optional): 
            Loss threshold (as a %) at which a position is automatically closed. 
            Default: NaN (disabled).

        take_profit (float > 0, optional): 
            Profit threshold (as a %) at which a position is automatically closed. 
            Default: NaN (disabled).

        close_all (bool, optional): 
            If True, forces the closing of all open positions at the end of the backtest period. 
            Default: True.

        transac_fees (float > 0, optional): 
            Transaction fee (for both buy and sell), expressed as a percentage. 
            Default: 0.001 (0.1%).

        borrow_rate (float > 0, optional): 
            Annual interest rate for borrowing assets (used in short positions). 
            Default: 0.02 (2%).

        borrowing_cash_fees (float, optional): 
            Annual fee for borrowing cash. 
            Default: 0.01 (1%).

        loan_cash_fees (float, optional): 
            Annual fee applied to cash lending. 
            Default: 0.0.

        maintenance_margin_pct (float, optional): 
            Maintenance margin percentage required when holding short positions (e.g., 1.3 = 130%). 
            Default: 1.3.

        annual_discount_rate (float, optional): 
            Annual discount rate used for financial calculations (e.g., NPV). 
            Default: 0.03 (3%).
        """


        validator = DataValidator(orders_df_input, prices_df_input, stop_loss, take_profit, bench_df_input)
        orders_df, prices_df, bench_df = validator.validate_all()
        orders_df.sort_values("Date", inplace=True)

        # Ajout automatique si nécessaire
        if "StopLoss" not in orders_df.columns:
            orders_df["StopLoss"] = stop_loss
        if "TakeProfit" not in orders_df.columns:
            orders_df["TakeProfit"] = take_profit

        self.orders_df = orders_df
        self.prices_df = prices_df
        self.bench_df = bench_df
        self.close_all = close_all
        self.transac_fees = transac_fees
        self.borrow_rate = borrow_rate
        self.borrowing_cash_fees = borrowing_cash_fees
        self.loan_cash_fees = loan_cash_fees
        self.maintenance_margin_pct = maintenance_margin_pct
        self.annual_discount_rate = annual_discount_rate
        self.base = base

        # Initialisation des objets de sortie
        self.builder = None
        self.cumulative_pnl_portfolio = None
        self.daily_pnl_portfolio = None
        self.portfolio_returns = None
        self.drawdown = None
        self.metrics = None

    def run(self):
        # Création du portefeuille
        self.builder = PortfolioBuilder(
            orders_df_input=self.orders_df,
            df_stock_prices=self.prices_df,
            close_all=self.close_all,
            transac_fees=self.transac_fees,
            borrow_rate=self.borrow_rate,
            base=self.base,
            maintenance_margin_pct=self.maintenance_margin_pct,
            borrowing_cash_fees=self.borrowing_cash_fees,
            loan_cash_fees=self.loan_cash_fees
        )
        self.builder.build_df()

        if not self.close_all :
            unclosed_positions = self.builder.df_valeur_portfolio.iloc[-1]
            last_day = unclosed_positions.name.date()
            nb_unclosed_positions = unclosed_positions.ne(0).sum()
            valeur_unclosed_positions = unclosed_positions.abs().sum()
            print(f"Au dernier jour ({last_day}), il y a {nb_unclosed_positions} positions not closed d'une valeur totale de {valeur_unclosed_positions:.2f}$.")


        self.df_ABP = calculate_avg_buying_price(self.builder.df_volume_order, self.builder.df_volume_portfolio, self.builder.df_stock_prices_reindexed)
        self.profit_long_positions, self.profit_short_positions = calculate_profit_transaction(self.builder.df_volume_portfolio, self.builder.df_stock_prices_reindexed, self.df_ABP)
        self.dic_winners_losers_long_short, self.dic_stats_operations = compute_metrics_per_ops(self.profit_long_positions, self.profit_short_positions)    

        # Calculs de performance
        self.cumulative_pnl_portfolio, self.daily_pnl_portfolio, self.daily_pnl_per_ticker = compute_portfolio_pnl(self.builder.df_volume_portfolio, self.builder.df_stock_prices_reindexed)
        
        max_cash_needed = self.builder.cash_consumption_with_costs.max()

        self.drawdown = compute_drawdown(self.cumulative_pnl_portfolio, max_cash_needed)
        self.portfolio_returns = compute_portfolio_returns(self.daily_pnl_portfolio, self.builder.df_valeur_portfolio)

        # Calcul des métriques finales
        self.portfolio_metrics = compute_metrics(
            self.portfolio_returns,
            self.cumulative_pnl_portfolio,
            self.drawdown,
            self.builder.df_volume_order,
            max_cash_needed,
            self.annual_discount_rate,
            self.base
        )

        # TODO: s'occuper de df_metrics_max_tickers
        self.df_metrics_per_ticker, self.df_metrics_max_tickers = metrics_tickers(self.builder.df_valeur_order, self.daily_pnl_per_ticker)

        # self.metrics = self.portfolio_metrics
        self.metrics = {**self.portfolio_metrics, **self.dic_stats_operations}

        # self.metrics.update(self.dic_stats_operations)

        if not self.bench_df.empty :
            self.dic_metrics_regression, self.coef_dict_regression = run_regression_factor_exposition(self.portfolio_returns, self.bench_df)
            self.bar_chart_factor_expo = plot_factor_exposition(self.coef_dict_regression)
            print(self.dic_metrics_regression)
            # self.bench_expo_summary = compute_factor_exposition(self.portfolio_returns, self.bench_df)
            # print(self.bench_expo_summary)

        # Création des graphs plotly
        # self.cash_consumption_graph = plot_cash_consumption(self.builder.cash_consumption_with_costs)
        self.cumulative_pnl_graph = plot_cumulative_pnl(self.cumulative_pnl_portfolio)
        self.drawdown_graph = plot_drawdown(self.drawdown)
        # TODO: something's wrong, the number do not add up
        # self.cash_graph = plot_cash_usage_breakdown(
        #     self.builder.cash_consumption_with_costs,
        #     self.builder.outflows_buy,
        #     self.builder.inflows_sell,
        #     self.builder.df_margin_call,
        #     self.builder.outflows_transac_fees,
        #     self.builder.outflows_repo,
        #     self.builder.cash_costs,
        #     self.builder.cash_gains,
        # )
        self.returns_histogram = plot_histo_returns_distrib(self.portfolio_returns)
        self.hit_ratio_pie = plot_pie_hit_ratio(self.dic_winners_losers_long_short)
        self.volume_vs_perf_scatter_plot = plot_volume_against_perf(self.df_metrics_per_ticker)


    def summary(self):
        if self.metrics is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Lance `run()` d'abord.")
        return self.metrics
    
    def plot_graphs(self) :
        # TODO: something's wrong, the number do not add up
        # self.cash_graph.show()
        self.cumulative_pnl_graph.show()
        self.drawdown_graph.show()
        self.returns_histogram.show()
        self.hit_ratio_pie.show()
        self.volume_vs_perf_scatter_plot.show()
        if not self.bench_df.empty :
            self.bar_chart_factor_expo.show()
