import pandas as pd

import os.path

# modules / classes
import modules.data_preparation as data_prepration
import modules.models as models

from tensorflow.keras.models import load_model


def main() -> None:
    n_steps_in, n_steps_out, n_features = 5, 1, 1
    epochs = 50
    n_splits = 10

    df = pd.read_csv("./data/stock_details_5_years.csv")

    print("\n")
    print("Company: AAPL")
    print("\n")
    AAPL_model = models.Timeseries(
        n_splits, n_steps_in, n_steps_out, n_features, epochs, "AAPL"
    )

    AAPL = data_prepration.DataPrep(df, "AAPL")

    df_AAPL, scaler_AAPL = AAPL.prepare_data()

    train_df_AAPL, test_df_AAPL = AAPL.train_test_split(df_AAPL)

    X_train_AAPL, y_train_AAPL = AAPL.split_sequence(
        train_df_AAPL["Close"], n_steps_in, n_steps_out
    )
    X_test_AAPL, y_test_AAPL = AAPL.split_sequence(
        test_df_AAPL["Close"], n_steps_in, n_steps_out
    )

    X_train_AAPL = X_train_AAPL.reshape(
        (X_train_AAPL.shape[0], X_train_AAPL.shape[1], n_features)
    )
    x_input_AAPL = X_test_AAPL.reshape(
        (X_test_AAPL.shape[0], X_test_AAPL.shape[1], n_features)
    )

    if os.path.isfile("./models/model_LSTM_AAPL.keras") is False:
        AAPL_model.models()
        AAPL_model.train_models(X_train_AAPL, y_train_AAPL)

        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_AAPL,
        ) = AAPL_model.test_prediction_if_no_previous_models(
            x_input_AAPL, df_AAPL, scaler_AAPL
        )
        AAPL_model.figure(df, test_df_AAPL, mean_LSTM, mean_Conv1D, mean_GRU)
    else:
        model_GRU = load_model("./models/model_GRU_AAPL.keras")
        model_Conv1D = load_model("./models/model_Conv1D_AAPL.keras")
        model_LSTM = load_model("./models/model_LSTM_AAPL.keras")

        print(model_GRU.summary(), model_Conv1D.summary(), model_LSTM.summary())
        (
            mean_LSTM,
            mean_Conv1D,
            mean_GRU,
            df_AAPL,
        ) = AAPL_model.test_prediction_if_previous_models(
            x_input_AAPL, df_AAPL, model_LSTM, model_Conv1D, model_GRU, scaler_AAPL
        )

        AAPL_model.figure(df_AAPL, test_df_AAPL, mean_LSTM, mean_Conv1D, mean_GRU)

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
