import glob
import json
import os
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
from loguru import logger
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import typer

from nps_estimator.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---------------- DEFAULT PATHS ----------------
    model_path: Path = MODELS_DIR / "model.joblib",
    features_path: Path = PROCESSED_DATA_DIR / "features.joblib",
    raw_df_path: Path = RAW_DATA_DIR / "desafio_nps_fase_1.csv",
    # -----------------------------------------------
):
    """
    Executa o pipeline de treinamento do modelo de regressão para estimativa de NPS.

    Esta função realiza as seguintes etapas:
    1. Carrega o dataset de NPS diretamente de um repositório no GitHub.
    2. Seleciona as variáveis relevantes para o treinamento do modelo.
    3. Divide os dados em conjuntos de treino e teste.
    4. Localiza o modelo LightGBM previamente otimizado mais recente e carrega
       seus hiperparâmetros.
    5. Treina um modelo `LGBMRegressor` com os parâmetros carregados.
    6. Avalia o modelo no conjunto de teste utilizando MAE, RMSE e R².
    7. Salva o modelo treinado no diretório de modelos.

    Args:
        model_path (Path, optional):
            Caminho onde o modelo treinado será salvo no formato `.joblib`.
            Por padrão, o modelo é salvo em `MODELS_DIR / "model.joblib"`.
        features_path (Path, optional):
            Caminho para o arquivo `.joblib` contendo a lista de colunas
            selecionadas para o treinamento. Por padrão, é `PROCESSED_DATA_DIR / "features.joblib"`.
        raw_df_path (Path, optional):
            Caminho para o arquivo CSV contendo o dataset bruto. Por padrão, é
            `RAW_DATA_DIR / "desafio_nps_fase_1.csv"`.

    Raises:
        typer.Exit:
            Caso nenhum modelo otimizado ou arquivo de parâmetros seja encontrado
            no diretório de modelos.

    Side Effects:
        - Realiza download de dados via HTTP.
        - Escreve arquivos de modelo no sistema de arquivos.
        - Emite logs informativos durante todas as etapas do pipeline.

    Returns:
        None
    """

    # -------------------- CODE --------------------
    logger.info("Loading dataframe")
    df = pd.read_csv(raw_df_path)

    columns2select = joblib.load(features_path)
    df = df[columns2select]

    # -------------- SPLIT DATASET - TRAIN|TEST ---------------
    logger.info("Separating train test data")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["nps_score"]), df["nps_score"], test_size=0.3, random_state=42
    )

    # -------------- LOAD MODEL BEST PARAMS ---------------
    logger.info("Load model params")
    # Listar todos os arquivos de modelo salvos
    model_files = sorted(
        glob.glob(os.path.join(MODELS_DIR, "lgbm_best_model_*.joblib")), reverse=True
    )

    if model_files:
        # Carregar o modelo mais recente
        latest_model_path = model_files[0]

        # Extrair timestamp do caminho
        model_timestamp = (
            os.path.basename(latest_model_path)
            .replace("lgbm_best_model_", "")
            .replace(".joblib", "")
        )

        # Carregar parâmetros
        params_file = os.path.join(MODELS_DIR, f"best_params_{model_timestamp}.json")
        with open(params_file, "r") as f:
            loaded_params = json.load(f)

    else:
        logger.error("✗ Nenhum modelo encontrado na pasta ../models/")
        raise typer.Exit(code=1)

    # -------------- TRAIN MODEL ---------------
    logger.info("Training model")
    model = LGBMRegressor(**loaded_params)
    model.fit(X_train, y_train)

    # Fazer predições com o modelo otimizado
    y_pred = model.predict(X_test)

    # Avaliar o modelo otimizado
    mae_best = mean_absolute_error(y_test, y_pred)
    rmse_best = root_mean_squared_error(y_test, y_pred)
    r2_best = r2_score(y_test, y_pred)

    logger.info("=== Resultados do Modelo Otimizado ===")
    logger.info(f"MAE: {mae_best:.4f}")
    logger.info(f"RMSE: {rmse_best:.4f}")
    logger.info(f"R²: {r2_best:.4f}")

    # -------------- SAVE MODEL ---------------
    logger.info("Saving model")

    # Salvar o modelo otimizado com compressão
    joblib.dump(model, model_path, compress=3)
    print(f"✓ Modelo salvo em: {model_path}")

    logger.success("Modeling training complete.")
    # -----------------------------------------------


if __name__ == "__main__":
    app()
