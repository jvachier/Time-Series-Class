import numpy as np
import pandas as pd

import os.path
from dataclasses import dataclass
import matplotlib.pylab as plt
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.layers import Conv1D, GRU
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


@dataclass(slots=True)
class Timeseries:
    n_splits: int
    n_steps_in: int
    n_steps_out: int
    n_features: int
    epochs: int
    company: str
    model_Conv1D: object = None
    model_GRU: object = None
    model_LSTM: object = None

    def models(self) -> None:
        self.model_Conv1D = Sequential()
        self.model_Conv1D.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_Conv1D.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.model_Conv1D.add(Flatten())
        self.model_Conv1D.add(Dense(self.n_steps_out))
        self.model_Conv1D.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

        self.model_GRU = Sequential()
        self.model_GRU.add(
            GRU(
                200,
                return_sequences=True,
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_GRU.add(GRU(200, return_sequences=True))
        self.model_GRU.add(GRU(200, return_sequences=True))
        self.model_GRU.add(GRU(200))
        self.model_GRU.add(Dense(self.n_steps_out))
        self.model_GRU.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

        self.model_LSTM = Sequential()
        self.model_LSTM.add(
            LSTM(
                200,
                return_sequences=True,
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_LSTM.add(LSTM(200, return_sequences=True))
        self.model_LSTM.add(LSTM(200, return_sequences=True))
        self.model_LSTM.add(LSTM(200))
        self.model_LSTM.add(Dense(self.n_steps_out))
        self.model_LSTM.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

    def train_models(self, X_train: np.array, y_train: list) -> None:
        scores_LSTM = []
        scores_Conv1D = []
        scores_GRU = []

        timefold = TimeSeriesSplit(self.n_splits).split(X_train, y_train)
        for k, (train, test) in enumerate(timefold):
            self.model_LSTM.fit(
                X_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_LSTM = self.model_LSTM.evaluate(
                X_train[test],
                y_train[test],
                verbose=0,
            )
            scores_LSTM.append(score_LSTM)
            print(
                "LSTM - Fold: %2d, Acc.: %.3f, Loss: %.3f"
                % (k + 1, score_LSTM[1], score_LSTM[0])
            )

            self.model_Conv1D.fit(
                X_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_Conv1D = self.model_Conv1D.evaluate(
                X_train[test],
                y_train[test],
                verbose=0,
            )
            scores_Conv1D.append(score_Conv1D)
            print(
                "Conv1D - Fold: %2d, Acc.: %.3f, Loss: %.3f"
                % (k + 1, score_Conv1D[1], score_Conv1D[0])
            )

            self.model_GRU.fit(
                X_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_GRU = self.model_GRU.evaluate(
                X_train[test],
                y_train[test],
                verbose=0,
            )
            scores_GRU.append(score_GRU)
            print(
                "GRU - Fold: %2d, Acc.: %.4f, Loss: %.4f"
                % (k + 1, score_GRU[1], score_GRU[0])
            )
        if (
            os.path.isfile("./models/model_LSTM_" + str(self.company) + ".keras")
            is False
        ):
            self.model_LSTM.save("./models/model_LSTM_" + str(self.company) + ".keras")
            self.model_Conv1D.save(
                "./models/model_Conv1D_" + str(self.company) + ".keras"
            )
            self.model_GRU.save("./models/model_GRU_" + str(self.company) + ".keras")

    def test_prediction_if_no_previous_models(
        self, x_input: np.array, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        yhat_LSTM = self.model_LSTM.predict(x_input, verbose=0)
        yhat_Conv1D = self.model_Conv1D.predict(x_input, verbose=0)
        yhat_GRU = self.model_GRU.predict(x_input, verbose=0)

        mean_LSTM = np.mean(yhat_LSTM, axis=1)
        mean_Conv1D = np.mean(yhat_Conv1D, axis=1)
        mean_GRU = np.mean(yhat_GRU, axis=1)

        mean_LSTM = self.scaler.inverse_transform(mean_LSTM.reshape(-1, 1)).reshape(
            len(mean_LSTM),
        )
        mean_Conv1D = self.scaler.inverse_transform(mean_Conv1D.reshape(-1, 1)).reshape(
            len(mean_Conv1D),
        )
        mean_GRU = self.scaler.inverse_transform(mean_GRU.reshape(-1, 1)).reshape(
            len(mean_GRU),
        )

        df["Close"] = self.scaler.inverse_transform(
            df["Close"].to_numpy().reshape(-1, 1)
        ).reshape(self.old_df)
        return mean_LSTM, mean_Conv1D, mean_GRU, df

    def test_prediction_if_previous_models(
        self,
        x_input: np.array,
        df: pd.DataFrame,
        model_LSTM: object,
        model_Conv1D: object,
        model_GRU: object,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        yhat_LSTM = model_LSTM.predict(x_input, verbose=0)
        yhat_Conv1D = model_Conv1D.predict(x_input, verbose=0)
        yhat_GRU = model_GRU.predict(x_input, verbose=0)

        mean_LSTM = np.mean(yhat_LSTM, axis=1)
        mean_Conv1D = np.mean(yhat_Conv1D, axis=1)
        mean_GRU = np.mean(yhat_GRU, axis=1)

        mean_LSTM = self.scaler.inverse_transform(mean_LSTM.reshape(-1, 1)).reshape(
            len(mean_LSTM),
        )
        mean_Conv1D = self.scaler.inverse_transform(mean_Conv1D.reshape(-1, 1)).reshape(
            len(mean_Conv1D),
        )
        mean_GRU = self.scaler.inverse_transform(mean_GRU.reshape(-1, 1)).reshape(
            len(mean_GRU),
        )

        df["Close"] = self.scaler.inverse_transform(
            df["Close"].to_numpy().reshape(-1, 1)
        ).reshape(self.old_df)
        return mean_LSTM, mean_Conv1D, mean_GRU, df

    def figure(
        self,
        df: pd.DataFrame,
        test_df: pd.DataFrame,
        LSTM: pd.DataFrame,
        Conv1D: pd.DataFrame,
        GRU: pd.DataFrame,
        n_steps_in: int,
    ) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], "--b", label=self.company)
        plt.plot(test_df["Date"][n_steps_in:], LSTM, "r", label="RNN using LSTM")
        plt.plot(test_df["Date"][n_steps_in:], Conv1D, "k", label="Conv1D")
        plt.plot(test_df["Date"][n_steps_in:], GRU, "g", label="GRU")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Close")
        if (
            os.path.isfile(
                "./figures/time_series_prediction_" + str(self.company) + ".png"
            )
            is False
        ):
            plt.savefig(
                "./figures/time_series_prediction_" + str(self.company) + ".png"
            )
        plt.show()
