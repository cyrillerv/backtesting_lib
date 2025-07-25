import pandas as pd
from .positions import PortfolioBuilder
from .utils import (
    compute_portfolio_pnl,
    compute_portfolio_returns,
    compute_drawdown
)
from .metrics import *
from .plotting import *

from .data_validator import DataValidator

class BacktestEngine:
    def __init__(self, 
                 orders_df_input, 
                 prices_df_input, 
                 bench_df_input=pd.DataFrame(),
                 transac_fees=0.001, 
                 borrow_rate=0.02, 
                 borrowing_cash_fees=0.01,
                 loan_cash_fees=0.0,
                 maintenance_margin_pct=1.3,
                 annual_discount_rate=0.03,
                 base=252):

        # TODO: inclure la vérification du df des benchmarks
        validator = DataValidator(orders_df_input, prices_df_input, bench_df_input)
        orders_df, prices_df, bench_df = validator.validate_all()


        self.orders_df = orders_df
        self.prices_df = prices_df
        self.bench_df = bench_df
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
            transac_fees=self.transac_fees,
            borrow_rate=self.borrow_rate,
            base=self.base,
            maintenance_margin_pct=self.maintenance_margin_pct,
            borrowing_cash_fees=self.borrowing_cash_fees,
            loan_cash_fees=self.loan_cash_fees
        )
        self.builder.build_df()

        # Calculs de performance
        self.cumulative_pnl_portfolio, self.daily_pnl_portfolio = compute_portfolio_pnl(self.builder.df_volume_portfolio, self.builder.df_stock_prices_reindexed)
        
        max_cash_needed = self.builder.cash_consumption_with_costs.max()

        self.drawdown = compute_drawdown(self.cumulative_pnl_portfolio, max_cash_needed)
        self.portfolio_returns = compute_portfolio_returns(self.daily_pnl_portfolio, self.builder.df_valeur_portfolio)

        # Calcul des métriques finales
        self.metrics = compute_metrics(
            self.portfolio_returns,
            self.cumulative_pnl_portfolio,
            self.drawdown,
            self.builder.df_volume_order,
            max_cash_needed,
            self.annual_discount_rate,
            self.base
        )

        if not self.bench_df.empty :
            self.bench_expo_summary = compute_factor_exposition(self.portfolio_returns, self.bench_df)
            print(self.bench_expo_summary)

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
