import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train(data_path: str = "inputs/dados.txt", experiment_name: str = "Previsao Sorvete"):
    """Treina um modelo simples (LinearRegression), registra no MLflow e retorna run_id, modelo e métricas."""
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(data_path)
    X = df[["temperatura"]]
    y = df["vendas"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = modelo.score(X_test, y_test)

    with mlflow.start_run() as run:
        mlflow.log_param("modelo", "LinearRegression")
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("r2", float(r2))

        # salvar o modelo no artefato do MLflow
        mlflow.sklearn.log_model(modelo, "modelo_sorvete")

        run_id = run.info.run_id

    print(f"Treino finalizado. run_id: {run_id}")
    print(f"Métricas: mse={mse:.4f}, r2={r2:.4f}")

    return run_id, modelo, {"mse": mse, "r2": r2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o modelo de previsão de sorvetes e registra no MLflow")
    parser.add_argument("--data_path", type=str, default="inputs/dados.txt")
    parser.add_argument("--experiment", type=str, default="Previsao Sorvete")
    args = parser.parse_args()

    train(data_path=args.data_path, experiment_name=args.experiment)
