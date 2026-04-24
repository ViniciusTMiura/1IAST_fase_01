from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
from loguru import logger
import pandas as pd
import typer

from nps_estimator.config import MODELS_DIR, PROCESSED_DATA_DIR

FILL_VALUE: float = 0.0

app = typer.Typer()


@lru_cache
def load_artifacts(model_path, features_path):
    """
    Carrega e cacheia os artefatos necessários para inferência do modelo de NPS.

    Esta função carrega do sistema de arquivos o modelo treinado e a lista de
    features utilizadas durante o treinamento. Para garantir consistência
    entre treino e inferência, a feature alvo (`nps_score`) é removida da lista,
    caso esteja presente.

    O uso do decorador `@lru_cache` evita leituras repetidas do disco,
    melhorando significativamente a performance em cenários de múltiplas
    chamadas (por exemplo, APIs ou pipelines de inferência).

    Args:
        model_path (Path or str):
            Caminho para o arquivo `.joblib` contendo o modelo treinado.

        features_path (Path or str):
            Caminho para o arquivo `.joblib` contendo a lista de features
            utilizadas pelo modelo.

    Returns:
        tuple:
            Uma tupla contendo:
            - model: Modelo treinado carregado em memória.
            - features (list[str]): Lista de nomes das features utilizadas
              na inferência, sem a variável alvo (`nps_score`).

    Raises:
        FileNotFoundError:
            Caso o modelo ou o arquivo de features não seja encontrado.

        Exception:
            Caso ocorra erro durante o carregamento dos artefatos.

    Side Effects:
        - Leitura de arquivos do sistema de arquivos.
        - Cache dos artefatos em memória para reutilização futura.
    """

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    if "nps_score" in features:
        features.remove("nps_score")
    return model, features


def predict_nps(
    data_input: Dict[str, Any],
    # ---------------- DEFAULT PATHS ----------------
    features_path: Path = PROCESSED_DATA_DIR / "features.joblib",
    model_path: Path = MODELS_DIR / "model.joblib",
    # -----------------------------------------------
) -> float:
    """
    Realiza inferência de NPS a partir de dados de entrada de um cliente.

    Esta função carrega um modelo previamente treinado e a lista de features
    utilizadas no treinamento, prepara os dados de entrada garantindo
    consistência com as features esperadas e retorna a predição do score de NPS.

    Caso alguma feature esperada pelo modelo não esteja presente nos dados
    de entrada, seu valor é preenchido com zero.

    Args:
        data_input (dict):
            Dicionário contendo os atributos do cliente necessários para
            a inferência. As chaves devem corresponder aos nomes das features
            utilizadas no treinamento do modelo.

        features_path (Path, optional):
            Caminho para o artefato `.joblib` contendo a lista de features
            utilizadas pelo modelo. Por padrão, utiliza
            `PROCESSED_DATA_DIR / "features.joblib"`.

        model_path (Path, optional):
            Caminho para o artefato `.joblib` do modelo treinado.
            Por padrão, utiliza `MODELS_DIR / "model.joblib"`.

    Returns:
        float:
            Valor do score do NPS para o registro.

    Raises:
        FileNotFoundError:
            Caso o modelo ou o arquivo de features não seja encontrado.

        ValueError:
            Caso os dados de entrada estejam em formato inválido.

    Side Effects:
        - Carrega artefatos do sistema de arquivos.
        - Emite logs informativos e avisos durante o processo de inferência.
    """
    # ---------------- VALIDATE INPUTS ---------------
    if not isinstance(data_input, dict):
        raise ValueError("data_input must be a dictionary")

    # ---------------- LOAD ARTIFACTS ---------------
    logger.info("Loading model and features...")
    model, features = load_artifacts(model_path, features_path)
    logger.info(f"Features used for inference: {features}")

    # ---------------- PROCESS INPUTS ---------------
    logger.info("Preparing input data for inference...")
    df_input = pd.DataFrame([data_input])

    df_inference = pd.DataFrame(columns=features)
    for feat in features:
        if feat in df_input.columns:
            df_inference[feat] = df_input[feat]
        else:
            logger.warning(f"Feature '{feat}' not found in input data. Filling with 0.")
            df_inference[feat] = FILL_VALUE
    # Mantendo a ordem das colunas
    df_inference = df_inference[features]

    # --------------- PERFORM INFERENCE ---------------
    logger.info("Performing inference...")
    predictions = model.predict(df_inference)

    return float(predictions[0])
    # -----------------------------------------


if __name__ == "__main__":
    # Exemplo: Cliente com atraso acima do ponto de ruptura
    cliente_teste = {
        "customer_age": 20,
        "customer_tenure_months": 10,
        "order_value": 1450.0,
        "discount_value": 140.0,
        "freight_value": 20.0,
        "delivery_attempts": 1,
        "payment_installments": 3,
        "items_quantity": 2,
        "delivery_time_days": 5,
        "delivery_delay_days": 4,
        "customer_service_contacts": 1,
        "complaints_count": 1,
        "resolution_time_days": 3,
    }

    res = predict_nps(cliente_teste)
    logger.info(f"Predicted NPS score for test client: {res:.2f}")
    logger.success("Inference complete.")
