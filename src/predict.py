import argparse
import mlflow
import mlflow.sklearn
import numpy as np


def predict_from_run(run_id: str, temperatura: float):
    model_uri = f"runs:/{run_id}/modelo_sorvete"
    modelo = mlflow.sklearn.load_model(model_uri)
    X = np.array([[float(temperatura)]])
    pred = modelo.predict(X)
    return float(pred[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz previsão usando um modelo registrado no MLflow")
    parser.add_argument("--run_id", type=str, required=True, help="ID do run no MLflow (apareceu após treino)")
    parser.add_argument("--temperatura", type=float, required=True, help="Temperatura para prever vendas")
    args = parser.parse_args()

    valor = predict_from_run(args.run_id, args.temperatura)
    print(f"Previsão de vendas para {args.temperatura}°C: {valor:.2f} unidades")
