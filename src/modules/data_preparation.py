import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler


@dataclass(slots=True)
class DataPrep:
    data: pd.DataFrame
    company: str

    def prepare_data(self) -> Tuple[pd.DataFrame, object]:
        scaler = MinMaxScaler()
        df_local = self.data[self.data["Company"] == self.company]
        df_local = df_local.dropna()
        df_local["Date"] = pd.to_datetime(df_local["Date"], utc=True)
        columns = df_local.columns
        df_local = df_local.drop(columns=columns[5:])
        cols = ["Close"]
        old_df = df_local[cols].shape
        df_local[cols] = scaler.fit_transform(
            df_local[cols].to_numpy().reshape(-1, 1)
        ).reshape(old_df)
        return df_local, scaler

    def split_sequence(
        self, sequence: pd.Series, n_steps_in: int, n_steps_out: int
    ) -> Tuple[np.array, np.array]:
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train_test_split(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_size = int(len(df) * 0.7)
        train_df, test_df = df[0:train_size], df[train_size : len(df)]
        return train_df, test_df
