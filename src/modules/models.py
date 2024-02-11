import os.path
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras.layers import GRU, LSTM, Conv1D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from sklearn.model_selection import TimeSeriesSplit


@dataclass(slots=True)
class Timeseries:
    n_splits: int
    n_steps_in: int
    n_steps_out: int
    n_features: int
    epochs: int
    company: str
    model_conv1d: Sequential = None
    model_gru: Sequential = None
    model_lstm: Sequential = None

    def models(self) -> None:
        self.model_conv1d = Sequential()
        self.model_conv1d.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_conv1d.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.model_conv1d.add(Flatten())
        self.model_conv1d.add(Dense(self.n_steps_out))
        self.model_conv1d.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

        self.model_gru = Sequential()
        self.model_gru.add(
            GRU(
                200,
                return_sequences=True,
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_gru.add(GRU(200, return_sequences=True))
        self.model_gru.add(GRU(200, return_sequences=True))
        self.model_gru.add(GRU(200))
        self.model_gru.add(Dense(self.n_steps_out))
        self.model_gru.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

        self.model_lstm = Sequential()
        self.model_lstm.add(
            LSTM(
                200,
                return_sequences=True,
                input_shape=(self.n_steps_in, self.n_features),
            )
        )
        self.model_lstm.add(LSTM(200, return_sequences=True))
        self.model_lstm.add(LSTM(200, return_sequences=True))
        self.model_lstm.add(LSTM(200))
        self.model_lstm.add(Dense(self.n_steps_out))
        self.model_lstm.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

    def train_models(self, x_train: np.array, y_train: list) -> None:
        scores_lstm = []
        scores_conv1d = []
        scores_gru = []

        timefold = TimeSeriesSplit(self.n_splits).split(x_train, y_train)
        for k, (train, test) in enumerate(timefold):
            self.model_lstm.fit(
                x_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_lstm = self.model_lstm.evaluate(
                x_train[test],
                y_train[test],
                verbose=0,
            )
            scores_lstm.append(score_lstm)
            print(
                f"Conv1D - Fold: ${(k + 1):.2d},"
                "Acc.: %${score_lstm[1]:.3f},"
                "Loss: ${score_lstm[0]:.3f}"
            )

            self.model_conv1d.fit(
                x_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_conv1d = self.model_conv1d.evaluate(
                x_train[test],
                y_train[test],
                verbose=0,
            )
            scores_conv1d.append(score_conv1d)
            print(
                f"Conv1D - Fold: ${(k + 1):.2d},"
                "Acc.: %${score_conv1d[1]:.3f},"
                "Loss: ${score_conv1d[0]:.3f}"
            )

            self.model_gru.fit(
                x_train[train],
                y_train[train],
                epochs=self.epochs,
                verbose=0,
                validation_split=0.3,
            )
            score_gru = self.model_gru.evaluate(
                x_train[test],
                y_train[test],
                verbose=0,
            )
            scores_gru.append(score_gru)
            print(
                f"Conv1D - Fold: ${(k + 1):.2d},"
                "Acc.: %${score_gru[1]:.4f},"
                "Loss: ${score_gru[0]:.4f}"
            )
        if (
            os.path.isfile("./models/model_LSTM_" + str(self.company) + ".keras")
            is False
        ):
            self.model_lstm.save("./models/model_LSTM_" + str(self.company) + ".keras")
            self.model_conv1d.save(
                "./models/model_Conv1D_" + str(self.company) + ".keras"
            )
            self.model_gru.save("./models/model_GRU_" + str(self.company) + ".keras")

    def test_prediction_if_no_previous_models(
        self,
        x_input: np.array,
        df: pd.DataFrame,
        scaler: object,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        yhat_lstm = self.model_lstm.predict(x_input, verbose=0)
        yhat_conv1d = self.model_conv1d.predict(x_input, verbose=0)
        yhat_gru = self.model_gru.predict(x_input, verbose=0)

        mean_lstm = np.mean(yhat_lstm, axis=1)
        mean_conv1d = np.mean(yhat_conv1d, axis=1)
        mean_gru = np.mean(yhat_gru, axis=1)

        mean_lstm = scaler.inverse_transform(mean_lstm.reshape(-1, 1)).reshape(
            len(mean_lstm),
        )
        mean_conv1d = scaler.inverse_transform(mean_conv1d.reshape(-1, 1)).reshape(
            len(mean_conv1d),
        )
        mean_gru = scaler.inverse_transform(mean_gru.reshape(-1, 1)).reshape(
            len(mean_gru),
        )
        old_df = df["Close"].shape
        df["Close"] = scaler.inverse_transform(
            df["Close"].to_numpy().reshape(-1, 1)
        ).reshape(old_df)
        return mean_lstm, mean_conv1d, mean_gru, df

    def test_prediction_if_previous_models(
        self,
        x_input: np.array,
        df: pd.DataFrame,
        model_lstm: object,
        model_conv1d: object,
        model_gru: object,
        scaler: object,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        yhat_lstm = model_lstm.predict(x_input, verbose=0)
        yhat_conv1d = model_conv1d.predict(x_input, verbose=0)
        yhat_gru = model_gru.predict(x_input, verbose=0)

        mean_lstm = np.mean(yhat_lstm, axis=1)
        mean_conv1d = np.mean(yhat_conv1d, axis=1)
        mean_gru = np.mean(yhat_gru, axis=1)

        mean_lstm = scaler.inverse_transform(mean_lstm.reshape(-1, 1)).reshape(
            len(mean_lstm),
        )
        mean_conv1d = scaler.inverse_transform(mean_conv1d.reshape(-1, 1)).reshape(
            len(mean_conv1d),
        )
        mean_gru = scaler.inverse_transform(mean_gru.reshape(-1, 1)).reshape(
            len(mean_gru),
        )

        old_df = df["Close"].shape
        df["Close"] = scaler.inverse_transform(
            df["Close"].to_numpy().reshape(-1, 1)
        ).reshape(old_df)
        return mean_lstm, mean_conv1d, mean_gru, df

    def figure(
        self,
        df: pd.DataFrame,
        test_df: pd.DataFrame,
        lstm: pd.DataFrame,
        conv1d: pd.DataFrame,
        gru: pd.DataFrame,
    ) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], "--b", label=self.company)
        plt.plot(test_df["Date"][self.n_steps_in :], lstm, "r", label="RNN using LSTM")
        plt.plot(test_df["Date"][self.n_steps_in :], conv1d, "k", label="Conv1D")
        plt.plot(test_df["Date"][self.n_steps_in :], gru, "g", label="GRU")
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
