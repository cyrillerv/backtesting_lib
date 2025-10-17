import pandas as pd
import numpy as np
import warnings

class DataValidator:
    def __init__(self, orders_df_input, stock_prices_df_input, stop_loss_input, take_profit_input, bench_df_input=None):
        self.orders_df = orders_df_input.copy()
        self.stock_prices_df = stock_prices_df_input.copy()
        self.stop_loss = stop_loss_input
        self.take_profit = take_profit_input
        self.bench_df = bench_df_input.copy()

    def check_columns_orders(self) :
        required_columns = {"Date", "Symbol", "Volume", "Type"}
        optional_columns = {"StopLoss", "TakeProfit"}

        input_columns = set(self.orders_df.columns)

        # Vérifie que toutes les colonnes obligatoires sont présentes
        missing_columns = required_columns - input_columns
        assert not missing_columns, \
            f"Colonnes obligatoires manquantes : {missing_columns}"

        # Vérifie qu'il n'y a pas de colonnes inconnues
        allowed_columns = required_columns.union(optional_columns)
        unknown_columns = input_columns - allowed_columns
        assert not unknown_columns, \
            f"Colonnes non reconnues dans le DataFrame : {unknown_columns}"

        # Conflit entre colonne StopLoss et argument stop_loss
        assert not ("StopLoss" in self.orders_df.columns and not np.isnan(self.stop_loss)), \
            "Vous ne pouvez pas à la fois passer une colonne 'StopLoss' et un argument stop_loss."

        # Conflit entre colonne TakeProfit et argument take_profit
        assert not ("TakeProfit" in self.orders_df.columns and not np.isnan(self.take_profit)), \
            "Vous ne pouvez pas à la fois passer une colonne 'TakeProfit' et un argument take_profit."



    def check_columns_content(self):

        # Convert types
        self.orders_df["Date"] = pd.to_datetime(self.orders_df["Date"], errors="raise")
        self.orders_df["Volume"] = self.orders_df["Volume"].astype(int)

        # Check Symbol is str
        assert self.orders_df["Symbol"].apply(lambda x: isinstance(x, str)).all(), "'Symbol' doit contenir uniquement des str"

        # Check Type values
        valid_types = {"Buy", "Sell"}
        invalid_types = set(self.orders_df["Type"]) - valid_types
        assert not invalid_types, f"Valeurs invalides dans 'Type' : {invalid_types}"

        # Check for NaN rows
        lines_with_NaN = self.orders_df[["Date", "Symbol", "Volume", "Type"]][self.orders_df.isnull().any(axis=1)]
        assert lines_with_NaN.empty, f"Des lignes avec NaN détectées :\n{lines_with_NaN}"
        
        return self.orders_df

    def check_stock_prices(self):
        # Check index type
        assert isinstance(self.stock_prices_df.index, pd.DatetimeIndex), "L'index n'est pas un DatetimeIndex"

        # Check all columns are float dtype
        non_float_cols = [col for col, dtype in self.stock_prices_df.dtypes.items() if not pd.api.types.is_float_dtype(dtype)]
        assert not non_float_cols, f"Ces colonnes ne sont pas de type float : {non_float_cols}"

        # Check columns all NaN
        cols_all_nan = self.stock_prices_df.columns[self.stock_prices_df.isna().all()]
        if not cols_all_nan.empty:
            raise ValueError(f"Les colonnes suivantes ne contiennent que des NaN : {list(cols_all_nan)}")

        # Replace inf values and warn
        cols_with_inf = self.stock_prices_df.columns[np.isinf(self.stock_prices_df).any()]
        if not cols_with_inf.empty:
            self.stock_prices_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            warnings.warn(f"Les colonnes suivantes contiennent des valeurs infinies : {list(cols_with_inf)}. Remplacées par NaN puis ffill.", stacklevel=2)
            self.stock_prices_df.ffill(inplace=True)

        # Fill remaining NaN
        self.stock_prices_df.ffill(inplace=True)
        self.stock_prices_df.bfill(inplace=True)

        return self.stock_prices_df
    
    def check_df_bench(self) :
        # Check index type
        assert isinstance(self.bench_df.index, pd.DatetimeIndex), "L'index n'est pas un DatetimeIndex"

        # Check all columns are float dtype
        non_float_cols = [col for col, dtype in self.bench_df.dtypes.items() if not pd.api.types.is_float_dtype(dtype)]
        assert not non_float_cols, f"Ces colonnes ne sont pas de type float : {non_float_cols}"

        # Check columns all NaN
        cols_all_nan = self.bench_df.columns[self.bench_df.isna().all()]
        if not cols_all_nan.empty:
            raise ValueError(f"Les colonnes suivantes ne contiennent que des NaN : {list(cols_all_nan)}")

        # Replace inf values and warn
        cols_with_inf = self.bench_df.columns[np.isinf(self.bench_df).any()]
        if not cols_with_inf.empty:
            self.bench_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            warnings.warn(f"Les colonnes suivantes contiennent des valeurs infinies : {list(cols_with_inf)}. Remplacées par NaN puis ffill.", stacklevel=2)
            self.bench_df.ffill(inplace=True)

        # Fill remaining NaN
        self.bench_df.ffill(inplace=True)
        self.bench_df.bfill(inplace=True)

        return self.bench_df

    def check_consistency(self):
        orders_symbols = set(self.orders_df["Symbol"].to_list())
        stock_symbols = set(self.stock_prices_df.columns)

        common_tickers = orders_symbols & stock_symbols
        missing_stock_prices = orders_symbols - stock_symbols

        if missing_stock_prices:
            print(f"Il manque les stockprices pour ces tickers: {missing_stock_prices}")

        # Keep only valid symbols
        self.orders_df = self.orders_df[self.orders_df["Symbol"].isin(common_tickers)]
        self.stock_prices_df = self.stock_prices_df.loc[:, list(common_tickers)]

        start_date = self.orders_df['Date'].min()
        last_order_date = self.orders_df['Date'].max()
        end_date = self.stock_prices_df.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        assert end_date >= last_order_date, f"Le df des stock prices ne va que jusqu'au {end_date} tandis que le dernier ordre est passé le {last_order_date}."
        self.stock_prices_df = self.stock_prices_df.reindex(full_date_range)
        self.stock_prices_df.ffill(inplace=True)
        if not self.bench_df.empty :
            assert self.bench_df.index.max() >= last_order_date, f"Le df des benchmarks ne va que jusqu'au {self.bench_df.index.max()} tandis que le dernier ordre est passé le {last_order_date}."
            self.bench_df = self.bench_df.reindex(full_date_range)
            self.bench_df.ffill(inplace=True)

        return self.orders_df, self.stock_prices_df, self.bench_df

    def validate_all(self):
        self.check_columns_orders()
        self.check_columns_content()
        self.check_stock_prices()
        if not self.bench_df.empty :
            self.check_df_bench()
        return self.check_consistency()
