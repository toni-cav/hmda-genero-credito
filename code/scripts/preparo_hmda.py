"""Prepara os dados compartilhados pelas etapas de modelagem HMDA."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


SEED = 444
WINSOR_LIMITS = (0.001, 0.999)
FEATURES_BASE = ["log_renda", "ltv", "dti", "loan_type", "state_code", "applicant_age"]
CATEGORICAL_FEATURES = ["dti", "loan_type", "state_code", "applicant_age"]
BASE_COLUMNS = [
    "state_code",
    "income",
    "combined_loan_to_value_ratio",
    "loan_type",
    "dti",
    "applicant_age",
    "negado",
    "mulher",
    
]


def project_root() -> Path:
    """Retorna a raiz do projeto."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"Diretório de dados não encontrado em {root / 'data'}")
    return root


def dividir_amostra(
    df: pd.DataFrame,
    test_size: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Divide a amostra, mantendo a estratificação por desfecho e gênero."""
    estrato = df["negado"].astype(str) + "_" + df["mulher"].astype(str)
    idx_treino, idx_teste = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=SEED,
        stratify=estrato,
    )
    return df.loc[idx_treino], df.loc[idx_teste], idx_treino, idx_teste


def criar_modelo_gradient_boosting(**parametros: int | float) -> HistGradientBoostingClassifier:
    """Cria o classificador usado nas versões com e sem gênero."""
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
        **parametros,
    )


def escolher_limiar(
    modelo: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    alvo: np.ndarray,
) -> float:
    """Escolhe o limiar com probabilidades de validação cruzada no treino."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilidades = cross_val_predict(
        clone(modelo), features, alvo, cv=cv, method="predict_proba"
    )[:, 1]
    return float(np.quantile(probabilidades, 1.0 - float(np.mean(alvo))))


def carregar_amostra_modelagem(
    caminho: Path | None = None,
    colunas_extras: Sequence[str] = (),
    amostra: int | None = None,
) -> pd.DataFrame:
    """Carrega e prepara a amostra completa usada nas etapas de modelagem."""
    arquivo = caminho or project_root() / "data" / "amostra_principal.parquet"
    colunas = list(dict.fromkeys([*BASE_COLUMNS, *colunas_extras]))
    df = pd.read_parquet(arquivo, columns=colunas)

    if amostra is not None:
        if amostra <= 0 or amostra > len(df):
            raise ValueError(f"A amostra deve estar entre 1 e {len(df):,}; recebido: {amostra}")
        df = df.sample(n=amostra, random_state=SEED)

    df = df.loc[df["income"] > 0]
    df = df.dropna(subset=["state_code", "applicant_age"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Nenhuma observação elegível em {arquivo}")

    lo, hi = df["combined_loan_to_value_ratio"].quantile(WINSOR_LIMITS)
    df["ltv"] = df["combined_loan_to_value_ratio"].clip(lo, hi)
    df["log_renda"] = np.log(df["income"])
    df["negado"] = df["negado"].astype("int8")
    df["mulher"] = df["mulher"].astype("int8")

    for coluna in CATEGORICAL_FEATURES:
        df[coluna] = df[coluna].astype(str).astype("category")

    return df
