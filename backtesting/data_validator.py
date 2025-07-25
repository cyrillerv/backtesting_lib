import pandas as pd
import numpy as np
import warnings

class DataValidator:
    def __init__(self, orders_df, stock_prices_df, bench_df_input=None):
        self.orders_df = orders_df.copy()
        self.stock_prices_df = stock_prices_df.copy()
        self.bench_df_input = bench_df_input.copy()
    
    def check_orders(self):
        expected_cols = {"Date", "Symbol", "Type", "Volume"}
        assert set(self.orders_df.columns) == expected_cols, f"Colonnes attendues : {expected_cols}"

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
        lines_with_NaN = self.orders_df[self.orders_df.isnull().any(axis=1)]
        assert lines_with_NaN.empty, f"Des lignes avec NaN détectées :\n{lines_with_NaN}"
        
        print("Orders dataframe check passed.")
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

        print("Stock prices dataframe check passed.")
        return self.stock_prices_df

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
        end_date = self.orders_df['Date'].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        self.stock_prices_df = self.stock_prices_df.reindex(full_date_range)
        self.stock_prices_df.ffill(inplace=True)

        print("Consistency check passed.")
        return self.orders_df, self.stock_prices_df, self.bench_df_input

    def validate_all(self):
        self.check_orders()
        self.check_stock_prices()
        return self.check_consistency()
