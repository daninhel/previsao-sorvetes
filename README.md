# 🍦 Previsão de Vendas de Sorvete

Projeto para prever vendas de sorvete com base na temperatura usando Machine Learning e registro com MLflow.

## Estrutura

previsao-sorvetes/

├── inputs/
│   └── dados.txt

├── notebooks/
│   └── modelo_sorvete.ipynb

├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── pipeline.py

├── README.md
├── requirements.txt
└── mlruns/  ← gerado automaticamente pelo MLflow

## Como rodar (Windows / PowerShell)

1. Criar e ativar um ambiente virtual (opcional mas recomendado):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Instalar dependências:

```powershell
pip install -r requirements.txt
```

3. Treinar o modelo e registrar no MLflow:

```powershell
python src\train_model.py --data_path inputs/dados.txt
```

O script imprimirá o `run_id` após o treino.

4. Iniciar a interface do MLflow (UI):

```powershell
mlflow ui
```

Abra http://localhost:5000 e veja os experimentos.

5. Fazer previsões (exemplo):

```powershell
python src\predict.py --run_id <run_id> --temperatura 30
```

## Observações
- O dataset é fictício e pequeno; serve para demonstração e testes locais.
- Próximos passos: testar outros modelos (RandomForest), adicionar validação cruzada e salvar artefatos (plots, métricas).
