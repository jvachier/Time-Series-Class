import os.path

import pandas as pd
from keras.models import load_model

from src.modules import data_preparation, models


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

    x_train_aapl, y_train_aapl = aapl.split_sequence(
        train_df_aapl["Close"], n_steps_in, n_steps_out
    )
    x_test_aapl, y_test_aapl = aapl.split_sequence(
        test_df_aapl["Close"], n_steps_in, n_steps_out
    )

    x_train_aapl = x_train_aapl.reshape(
        (x_train_aapl.shape[0], x_train_aapl.shape[1], n_features)
    )
    x_input_aapl = x_test_aapl.reshape(
        (x_test_aapl.shape[0], x_test_aapl.shape[1], n_features)
    )

    if os.path.isfile("./models/model_LSTM_AAPL.keras") is False:
        aapl_model.models()
        aapl_model.train_models(x_train_aapl, y_train_aapl)

        (
            mean_lstm,
            mean_conv1d,
            mean_gru,
            df_aapl,
        ) = aapl_model.test_prediction_if_no_previous_models(
            x_input_aapl, df_aapl, scaler_aapl
        )
        aapl_model.figure(df, test_df_aapl, mean_lstm, mean_conv1d, mean_gru)
    else:
        model_gru = load_model("./models/model_GRU_AAPL.keras")
        model_conv1d = load_model("./models/model_Conv1D_AAPL.keras")
        model_lstm = load_model("./models/model_LSTM_AAPL.keras")

        print(model_gru.summary(), model_conv1d.summary(), model_lstm.summary())
        (
            mean_lstm,
            mean_conv1d,
            mean_gru,
            df_aapl,
        ) = aapl_model.test_prediction_if_previous_models(
            x_input_aapl, df_aapl, model_lstm, model_conv1d, model_gru, scaler_aapl
        )

        aapl_model.figure(df_aapl, test_df_aapl, mean_lstm, mean_conv1d, mean_gru)

    print("\n")
    print("Company: ARES")
    print("\n")

    ares_model = models.Timeseries(
        n_splits, n_steps_in, n_steps_out, n_features, epochs, "ARES"
    )
    ares = data_preparation.DataPrep(df, "ARES")

    df_ares, scaler_ares = ares.prepare_data()
    train_df_ares, test_df_ares = ares.train_test_split(df_ares)

    x_train_ares, y_train_ares = ares.split_sequence(
        train_df_ares["Close"], n_steps_in, n_steps_out
    )
    x_test_ares, y_test_ares = ares.split_sequence(
        test_df_ares["Close"], n_steps_in, n_steps_out
    )

    x_train_ares = x_train_ares.reshape(
        (x_train_ares.shape[0], x_train_ares.shape[1], n_features)
    )
    x_test_ares = x_test_ares.reshape(
        (x_test_ares.shape[0], x_test_ares.shape[1], n_features)
    )

    if os.path.isfile("./models/model_LSTM_ARES.keras") is False:
        ares_model.models()
        ares_model.train_models(x_train_ares, y_train_ares)

        (
            mean_lstm,
            mean_conv1d,
            mean_gru,
            df_ares,
        ) = ares_model.test_prediction_if_no_previous_models(
            x_test_ares, df_ares, scaler_ares
        )
        ares_model.figure(df_ares, test_df_ares, mean_lstm, mean_conv1d, mean_gru)
    else:
        model_gru = load_model("./models/model_GRU_ARES.keras")
        model_conv1d = load_model("./models/model_Conv1D_ARES.keras")
        model_lstm = load_model("./models/model_LSTM_ARES.keras")
        print(model_gru.summary(), model_conv1d.summary(), model_lstm.summary())
        (
            mean_lstm,
            mean_conv1d,
            mean_gru,
            df_ares,
        ) = ares_model.test_prediction_if_previous_models(
            x_test_ares, df_ares, model_lstm, model_conv1d, model_gru, scaler_ares
        )
        ares_model.figure(df_ares, test_df_ares, mean_lstm, mean_conv1d, mean_gru)


if __name__ == "__main__":
    main()
