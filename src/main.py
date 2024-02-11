import os.path
import pandas as pd

from keras.models import load_model

from src.modules import data_preparation
from src.modules import models

def main() -> None:
    n_steps_in, n_steps_out, n_features = 5, 1, 1
    epochs = 50
    n_splits = 10

    df = pd.read_csv("./data/stock_details_5_years.csv")

    print("\n")
    print("Company: AAPL")
    print("\n")
    aapl_model = models.Timeseries(
        n_splits, n_steps_in, n_steps_out, n_features, epochs, "AAPL"
    )

    aapl = data_preparation.DataPrep(df, "AAPL")

    df_aapl, scaler_aapl = aapl.prepare_data()

    train_df_aapl, test_df_aapl = aapl.train_test_split(df_aapl)

    X_train_aapl, y_train_aapl = aapl.split_sequence(
        train_df_aapl["Close"], n_steps_in, n_steps_out
    )
    X_test_aapl, y_test_aapl = aapl.split_sequence(
        test_df_aapl["Close"], n_steps_in, n_steps_out
    )

    X_train_aapl = X_train_aapl.reshape(
        (X_train_aapl.shape[0], X_train_aapl.shape[1], n_features)
    )
    x_input_aapl = X_test_aapl.reshape(
        (X_test_aapl.shape[0], X_test_aapl.shape[1], n_features)
    )

    if os.path.isfile("./models/model_LSTM_AAPL.keras") is False:
        aapl_model.models()
        aapl_model.train_models(X_train_aapl, y_train_aapl)

        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_aapl,
        ) = aapl_model.test_prediction_if_no_previous_models(
            x_input_aapl, df_aapl, scaler_aapl
        )
        aapl_model.figure(df, test_df_aapl, mean_LSTM, mean_Conv1D, mean_GRU)
    else:
        model_GRU = load_model("./models/model_GRU_AAPL.keras")
        model_Conv1D = load_model("./models/model_Conv1D_AAPL.keras")
        model_LSTM = load_model("./models/model_LSTM_AAPL.keras")

        print(model_GRU.summary(), model_Conv1D.summary(), model_LSTM.summary())
        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_aapl,
        ) = aapl_model.test_prediction_if_previous_models(
            x_input_aapl, df_aapl, model_LSTM, model_Conv1D, model_GRU, scaler_aapl
        )

        aapl_model.figure(df_aapl, test_df_aapl, mean_LSTM, mean_Conv1D, mean_GRU)

    print("\n")
    print("Company: ARES")
    print("\n")

    ARES_model = models.Timeseries(
        n_splits, n_steps_in, n_steps_out, n_features, epochs, "ARES"
    )
    ARES = data_prepration.DataPrep(df, "ARES")

    df_ARES, scaler_ARES = ARES.prepare_data()
    train_df_ARES, test_df_ARES = ARES.train_test_split(df_ARES)

    X_train_ARES, y_train_ARES = ARES.split_sequence(
        train_df_ARES["Close"], n_steps_in, n_steps_out
    )
    X_test_ARES, y_test_ARES = ARES.split_sequence(
        test_df_ARES["Close"], n_steps_in, n_steps_out
    )

    X_train_ARES = X_train_ARES.reshape(
        (X_train_ARES.shape[0], X_train_ARES.shape[1], n_features)
    )
    x_input_ARES = X_test_ARES.reshape(
        (X_test_ARES.shape[0], X_test_ARES.shape[1], n_features)
    )

    if os.path.isfile("./models/model_LSTM_ARES.keras") is False:
        ARES_model.models()
        ARES_model.train_models(X_train_ARES, y_train_ARES)

        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_ARES,
        ) = ARES_model.test_prediction_if_no_previous_models(
            x_input_ARES, df_ARES, scaler_ARES
        )
        ARES_model.figure(df_ARES, test_df_ARES, mean_LSTM, mean_Conv1D, mean_GRU)
    else:
        model_GRU = load_model("./models/model_GRU_ARES.keras")
        model_Conv1D = load_model("./models/model_Conv1D_ARES.keras")
        model_LSTM = load_model("./models/model_LSTM_ARES.keras")
        print(model_GRU.summary(), model_Conv1D.summary(), model_LSTM.summary())
        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_ARES,
        ) = ARES_model.test_prediction_if_previous_models(
            x_input_ARES, df_ARES, model_LSTM, model_Conv1D, model_GRU, scaler_ARES
        )
        ARES_model.figure(df_ARES, test_df_ARES, mean_LSTM, mean_Conv1D, mean_GRU)


if __name__ == "__main__":
    main()
