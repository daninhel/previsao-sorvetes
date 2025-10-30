from train_model import train
import mlflow
import mlflow.sklearn


def run_pipeline():
    print("Iniciando pipeline: treino -> avaliação -> previsão de exemplo")
    run_id, modelo, metrics = train()

    # carregar modelo salvo no run e fazer uma previsão de exemplo
    model_uri = f"runs:/{run_id}/modelo_sorvete"
    modelo_carregado = mlflow.sklearn.load_model(model_uri)

    temperatura_exemplo = 30.0
    pred = modelo_carregado.predict([[temperatura_exemplo]])
    print(f"Previsão de vendas para {temperatura_exemplo}°C: {pred[0]:.2f}")


if __name__ == "__main__":
    run_pipeline()
