from os.path import join as join

import joblib
import mlflow
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from models.models import create_nn_model, model_predict, train_model
from modules.evaluate import evaluate_performance
from modules.preprocess import preprocessing, split
from modules.print_draw import draw_loss, print_data

# MLflow init
mlflow.set_experiment("Model performance")

base_model_path = join("models", "model_2026_03.pkl")
base_data_path = join("data", "df_new.csv")

with mlflow.start_run():
    mlflow.log_param("base_data_path", base_data_path)

    # Chargement des datasets
    df = pd.read_csv(base_data_path)

    # preprocesser les data
    X, y, _ = preprocessing(df)

    # split data in train and test dataset
    X_train, X_test, y_train, y_test = split(X, y)

    # create a new model
    # model = create_nn_model(X_train.shape[1])

    # sauvegarder le nouveau modèle
    # joblib.dump(model, base_model_path)

    # earlystopping callback
    # early_stop = EarlyStopping(
    #     monitor="val_loss", patience=10, restore_best_weights=True
    # )

    # entraîner le modèle
    loaded_model = joblib.load(base_model_path)
    mlflow.sklearn.log_model(
        loaded_model, name=base_model_path.split("/")[1].replace(".pkl", "")
    )

    # model, hist = train_model(
    #     loaded_model,
    #     X_train,
    #     y_train,
    #     X_val=X_test,
    #     y_val=y_test,
    #     epochs=10000,
    #     callbacks=[early_stop],
    # )

    # sauvegarder le modèle entrainé
    # joblib.dump(model, base_model_path)

    # Graphique directement dans mlflow
    # draw_loss(hist)

    # charger le modèle
    # loaded_model = joblib.load(base_model_path)

    # predire sur les valeurs de train
    y_pred = model_predict(loaded_model, X_train)

    # mesurer les performances MSE, MAE et R²
    perf = evaluate_performance(y_train, y_pred)
    print_data(perf, exp_name="exp_1 train")

    # predire sur les valeurs de tests
    y_pred = model_predict(loaded_model, X_test)

    # mesurer les performances MSE, MAE et R²
    perf = evaluate_performance(y_test, y_pred)
    print_data(perf, exp_name="exp_1 test")

    mlflow.log_metric("MSE", perf["MSE"])
    mlflow.log_metric("MAE", perf["MAE"])
    mlflow.log_metric("R²", perf["R²"])

    # WARNING ZONE on test d'entrainer le modèle plus longtemps mais sur les mêmes données
    # model2, hist2 = train_model(
    #     model_2024_08, X_train, y_train, X_val=X_test, y_val=y_test
    # )

    # y_pred = model_predict(model_2024_08, X_test)
    # perf = evaluate_performance(y_test, y_pred)

    # print_data(perf, exp_name="exp 2")
    # draw_loss(hist2)

    # mlflow.log_metric("MSE", perf["MSE"])
    # mlflow.log_metric("MAE", perf["MAE"])
    # mlflow.log_metric("R²", perf["R²"])
