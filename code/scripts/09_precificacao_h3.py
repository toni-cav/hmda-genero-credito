#!/usr/bin/env python3
"""Etapa G (H3): análise dos custos do crédito com efeitos fixos."""

import gc
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm

from preparo_hmda import project_root

RAIZ = project_root()
ARQUIVO = RAIZ / "data" / "hmda_2025.parquet"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
COLUNAS = [
    "action_taken", "lei", "state_code", "derived_sex", "interest_rate",
    "rate_spread", "total_loan_costs", "income", "loan_amount",
    "combined_loan_to_value_ratio", "debt_to_income_ratio", "loan_type",
    
]
DESFECHOS = ["rate_spread", "interest_rate", "total_loan_costs"]
CONTROLES = ["log_income", "log_loan_amount", "ltv", "dti_cat", "loan_type_cat"]
TERMOS_CONTROLE = ["log_income", "log_loan_amount", "ltv", "C(dti_cat)", "C(loan_type_cat)"]
Z95 = norm.ppf(0.975)


def carregar_dados():
    # Filtra no Parquet para carregar apenas os pedidos relevantes.
    df = pd.read_parquet(
        # action_taken está armazenada como texto neste Parquet.
        ARQUIVO, columns=COLUNAS, filters=[("action_taken", "==", "1")]
    )
    df = df.loc[df["derived_sex"].isin(["Male", "Female"])].copy()
    df["mulher"] = (df["derived_sex"] == "Female").astype("float32")
    for col in ["income", "loan_amount", "combined_loan_to_value_ratio", *DESFECHOS]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["log_income"] = np.log(df["income"].where(df["income"] > 0)).astype("float32")
    df["log_loan_amount"] = np.log(df["loan_amount"].where(df["loan_amount"] > 0)).astype("float32")
    df["ltv"] = df["combined_loan_to_value_ratio"]
    df["dti_cat"] = df["debt_to_income_ratio"].astype("category")
    df["loan_type_cat"] = df["loan_type"].astype("category")
    for col in ["lei", "state_code"]:
        df[col] = df[col].astype("category")
    usar = ["lei", "state_code", "mulher", *CONTROLES, *DESFECHOS]
    df = df[usar]
    del usar
    gc.collect()
    return df


def estimar(df):
    saida = []
    formula_base = "mulher + " + " + ".join(TERMOS_CONTROLE) + " | state_code + lei"
    for desfecho in DESFECHOS:
        colunas = [desfecho, "mulher", *CONTROLES, "state_code", "lei"]
        dados = df[colunas].dropna().reset_index(drop=True)
        formula = f"{desfecho} ~ {formula_base}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = pf.feols(formula, data=dados, vcov={"CRV1": "lei"})
        beta = float(modelo.coef()["mulher"])
        ep = float(modelo.se()["mulher"])
        saida.append({
            "desfecho": desfecho,
            "n_obs": int(modelo._N),
            "n_instituicoes": int(dados["lei"].nunique()),
            "coef_mulher": beta,
            "ep_cluster_lei": ep,
            "ic95_inf": beta - Z95 * ep,
            "ic95_sup": beta + Z95 * ep,
            "p_valor": float(2 * norm.sf(abs(beta / ep))),
            "formula": formula,
        })
        del modelo, dados
        gc.collect()
    return pd.DataFrame(saida)


def salvar(res):
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)
    res.to_csv(RESULTADOS / "etapa_09_precificacao_h3.csv", index=False)
    (RESULTADOS / "etapa_09_precificacao_h3.json").write_text(
        json.dumps(res.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(res))
    ax.errorbar(res["coef_mulher"], y,
                xerr=[res["coef_mulher"] - res["ic95_inf"], res["ic95_sup"] - res["coef_mulher"]],
                fmt="o", capsize=4, color="#245a81")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, res["desfecho"])
    ax.set_xlabel("Diferença estimada para mulheres (IC 95%)")
    ax.set_title("Etapa G — precificação do crédito")
    fig.tight_layout()
    fig.savefig(FIGURAS / "etapa_09_precificacao_h3.png", dpi=300)
    plt.close(fig)


def main():
    df = carregar_dados()
    print(f"Observações após action_taken=1 e gênero identificado: {len(df):,}")
    res = estimar(df)
    salvar(res)
    print(res.to_string(index=False))
    print("Arquivos gerados: resultados/etapa_09_precificacao_h3.csv, resultados/etapa_09_precificacao_h3.json, figuras/etapa_09_precificacao_h3.png")
    del df, res
    gc.collect()


if __name__ == "__main__":
    main()
