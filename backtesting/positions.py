# Outils pour avg_buy_price, PnL, cash, etc.
import numpy as np
import pandas as pd
from .utils import simulate_SL_TP_row

class PortfolioBuilder() :
    """
    Classe permettant de construire les principaux DataFrames liés à un backtest :
    - Valeur portefeuille
    - Frais de transaction
    - Coût des shorts
    - Appels de marge
    - Consommation de cash
    """
    def __init__(self, 
                 orders_df_input, 
                 df_stock_prices,
                 close_all,
                 transac_fees,
                 borrow_rate,
                 base,
                 maintenance_margin_pct,
                 borrowing_cash_fees,
                 loan_cash_fees
                 ) :
        self.orders_df_input = orders_df_input
        self.df_stock_prices = df_stock_prices
        self.close_all = close_all
        self.transac_fees = transac_fees
        self.borrow_rate = borrow_rate
        self.base = base
        self.maintenance_margin_pct = maintenance_margin_pct
        self.borrowing_cash_fees = borrowing_cash_fees
        self.loan_cash_fees = loan_cash_fees



    def calc_SL_TP(self) :
        # TODO: montrer ce que les SL ont permis d'éviter comme perte et ce le manque à gagner des TP. Montrer les bénéfices du SL et du TP sur le cash disponible.
        # self.orders_df_input.sort_values("Date", inplace=True)
        orders_df_with_SL_TP = self.orders_df_input.copy()
        orders_df_with_SL_TP['Flows'] = orders_df_with_SL_TP["Volume"] * np.where(
            orders_df_with_SL_TP["Type"] == "Buy", 1, -1
        )

        ## METHOD: FIFO
        for _, row in self.orders_df_input.dropna(how='all').iterrows():
            result = simulate_SL_TP_row(row, self.df_stock_prices, orders_df_with_SL_TP)
            # A chaque fois on envoye un nouveau df d'ordres en input car il fatu prendre en compte les précédents SL et TP
            if result is not None:
                orders_df_with_SL_TP = pd.concat([orders_df_with_SL_TP, pd.DataFrame([result])], ignore_index=True)

        # (Optionnel) trier pour faciliter la lecture
        orders_df_with_SL_TP = orders_df_with_SL_TP.sort_values(by="Date").reset_index(drop=True)
        self.orders_df_with_SL_TP = orders_df_with_SL_TP


    # On crée le df des volumes en portefeuille pour chaque date pour chaque ticker
    def build_df_portfolio_value(self) :
        """
        Construit le df qui contient la valeur de la position (currency du df en input) pour chaque ticker à chaque date.
        """
        # On crée le df des volumes en portefeuille pour chaque date pour chaque ticker
        orders_df = self.order_df_final.copy()
        orders_df["Sens_position"] = orders_df["Type"].map({"Buy": 1, "Sell": -1})
        orders_df["Volume_portfolio"] = orders_df["Sens_position"] * orders_df["Volume"]
        # Affiche le volume de transaction aux dates de transaction
        df_volume_order = orders_df.pivot_table(index="Date", columns="Symbol", values="Volume_portfolio", aggfunc='sum')
        self.df_volume_portfolio = df_volume_order.fillna(0).cumsum()

        # Crée un index de dates continues (jours ouvrés par exemple)
        full_index = pd.date_range(start=self.df_volume_portfolio.index.min(),
                                end=self.df_stock_prices.index.max(),
                                freq='D')  # 'B' = business days
        # Réindexe et forward fill
        self.df_volume_portfolio = self.df_volume_portfolio.reindex(full_index, method='ffill')

        self.df_volume_portfolio.fillna(0, inplace=True)

        self.df_stock_prices_reindexed = self.df_stock_prices.reindex(self.df_volume_portfolio.index, method='ffill').copy()
        

        df_valeur_portfolio = self.df_volume_portfolio * self.df_stock_prices_reindexed
        df_valeur_portfolio.fillna(0, inplace=True)

        self.df_volume_order = df_volume_order
        self.df_valeur_portfolio = df_valeur_portfolio

    def calculate_transac_fees(self) :

        df_valeur_order = self.df_volume_order * self.df_stock_prices
        # On passe en abs car on a des fees sur les long et les shorts
        df_transac_fees = (df_valeur_order.abs() * self.transac_fees)

        self.df_valeur_order = df_valeur_order
        self.df_transac_fees = df_transac_fees


    def build_df_repo_costs(self) :
        # On va calculer les frais dû au repo (pour les shorts pour l'instant)
        # On ne garde que les shorts
        df_valeur_portfolio_short = self.df_valeur_portfolio[self.df_valeur_portfolio < 0].copy()
        df_borrowing_stock_costs = df_valeur_portfolio_short * -self.borrow_rate / self.base

        self.df_valeur_portfolio_short = df_valeur_portfolio_short
        self.df_borrowing_stock_costs = df_borrowing_stock_costs


    # def build_df_repo_gains(df_valeur_portfolio, loan_rate, base=252) :
    #     """Ce qui est compliqué avec cette fonction c'est que si on implémente ça, ça libère du cash lorsqu'on prête nos actions => faut en tenir compte."""
    #     # On va calculer les gains dû au repo (si on prête les actions qu'on a en portefeuille.)
    #     # On ne garde que les long
    #     df_valeur_portfolio_long = df_valeur_portfolio[df_valeur_portfolio > 0].copy()
    #     df_loan_stock_gains = df_valeur_portfolio_long * loan_rate / base
    #     return df_loan_stock_gains


    def calculate_collat_short(self) :    
        # On va calculer les appels de marge, le cash bloqué avec les shorts, le collat
        # maintenance_margin_pct = 1.3 # 130%
        df_collat_short = self.df_valeur_portfolio_short * -self.maintenance_margin_pct
        df_collat_short.fillna(0, inplace=True)
        df_margin_call = df_collat_short.diff()

        df_margin_call + self.df_valeur_order

        self.df_margin_call = df_margin_call

    def calc_portfolio_flows(self) :
        # Lorsqu'on long ou qu'on close un short.
        self.outflows_buy = self.df_valeur_order.clip(lower=0).fillna(0)
        # Lorsqu'on short ou qu'on close un long
        self.inflows_sell = self.df_valeur_order.clip(upper=0).fillna(0)
        self.outflows_margin_call = self.df_margin_call.clip(lower=0).fillna(0)
        self.inflows_margin_call = self.df_margin_call.clip(upper=0).fillna(0)
        self.outflows_transac_fees = self.df_transac_fees.fillna(0)
        self.outflows_repo = self.df_borrowing_stock_costs.fillna(0)

        # Flux total par jour
        flows_portfolio = (
            self.outflows_buy +
            self.inflows_sell +
            self.outflows_margin_call +
            self.inflows_margin_call +
            self.outflows_transac_fees +
            self.outflows_repo
        )
        # Cumul du cash utilisé
        self.cumulative_flows = flows_portfolio.sum(axis=1).cumsum()

        self.cash_costs = self.cumulative_flows.clip(lower=0) * self.borrowing_cash_fees / self.base
        self.cash_gains = self.cumulative_flows.clip(upper=0) * self.loan_cash_fees / self.base

        self.cash_consumption_with_costs = self.cumulative_flows + self.cash_costs + self.cash_gains
        
    
    def build_df(self) :
        self.calc_SL_TP()
        self.order_df_final = self.orders_df_with_SL_TP
        self.build_df_portfolio_value()
        if self.close_all :
            # On regarde les positions qui n'ont jamais été clôturées.
            self.unclosed_positions = self.df_volume_portfolio.iloc[-1].replace(0, np.nan).dropna()
            # Puis on construit le df d'ordres pour les clôturer au dernier jour du backtest.
            self.unclosed_positions_orders = self.unclosed_positions.rename("Volume").to_frame().copy()
            self.unclosed_positions_orders.reset_index(inplace=True)
            self.unclosed_positions_orders['Date'] = self.unclosed_positions.name
            self.unclosed_positions_orders["Type"] = np.where(self.unclosed_positions_orders["Volume"] > 0, "Sell", "Buy")
            self.unclosed_positions_orders["Volume"] = self.unclosed_positions_orders["Volume"].abs()
            self.unclosed_positions_orders["Trigger"] = "AutomaticClosing"
            self.order_df_final = pd.concat([self.orders_df_with_SL_TP, self.unclosed_positions_orders], axis=0)
            # Et on relance la construction des df avec le df d'ordres qui inclut la clôture automatiques des ordres.
            self.build_df_portfolio_value()

        self.calculate_transac_fees()
        self.build_df_repo_costs()
        self.calculate_collat_short()
        self.calc_portfolio_flows()

    def get_results(self):
        return {
            "df_valeur_portfolio": self.df_valeur_portfolio,
            "df_volume_order": self.df_volume_order,
            "df_transac_fees": self.df_transac_fees,
            "df_borrowing_stock_costs": self.df_borrowing_stock_costs,
            "df_valeur_portfolio_short": self.df_valeur_portfolio_short,
            "df_margin_call": self.df_margin_call,
            "cash_consumption_with_costs": self.cash_consumption_with_costs,
        }



