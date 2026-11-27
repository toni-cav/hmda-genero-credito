#!/usr/bin/env python3
"""Etapa B (H1): regressão logística da negação de crédito por gênero."""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.special import expit
from scipy.stats import norm

from preparo_hmda import SEED as SEMENTE, project_root

# Caminhos e parâmetros

RAIZ = project_root()
ARQUIVO_DADOS = RAIZ / "data" / "amostra_principal.parquet"
PASTA_SAIDA = RAIZ / "resultados"

AMOSTRA_TESTE = None
Z95 = norm.ppf(0.975)
WINSOR_LTV = (0.001, 0.999)

COLUNAS = [
    "lei", "state_code", "income", "combined_loan_to_value_ratio",
    "loan_type", "dti", "applicant_age", "negado", "mulher",
]

# Primeiro item de cada lista é a categoria de referência
DTI_NIVEIS = ["<20%", "20%-<30%", "30%-<36%", "36%-<50%", "50%-60%", ">60%"]
TIPO_NIVEIS = ["1", "2", "3", "4"]
IDADE_NIVEIS = ["25-34", "<25", "35-44", "45-54", "55-64", "65-74", ">74"]


# 1. Preparação dos dados

def _registrar_corte(df_corte: pd.DataFrame, etapa: str, n_restante: int) -> dict:
    n = len(df_corte)
    return {
        "etapa": etapa,
        "removidas": n,
        "negacoes_removidas": int(df_corte["negado"].sum()),
        "taxa_negacao_removidas": float(df_corte["negado"].mean()) if n else np.nan,
        "pct_mulher_removidas": float(df_corte["mulher"].mean()) if n else np.nan,
        "n_restante": n_restante,
    }


def preparar_dados(caminho: Path, amostra: int | None = None):
    df = pd.read_parquet(caminho, columns=COLUNAS)
    if amostra:
        df = df.sample(n=amostra, random_state=SEMENTE).reset_index(drop=True)

    for col in ["dti", "loan_type"]:
        df[col] = df[col].astype(str)
    for col in ["lei", "state_code"]:
        df[col] = df[col].astype("string")
    for col in ["negado", "mulher"]:
        df[col] = df[col].astype(float)

    # Valida categorias antes de criar dummies
    for col, niveis in [("dti", DTI_NIVEIS), ("loan_type", TIPO_NIVEIS)]:
        extras = set(df[col].unique()) - set(niveis)
        if extras:
            raise ValueError(f"Categorias inesperadas em {col}: {sorted(extras)}")
    extras_idade = set(df["applicant_age"].dropna().unique()) - set(IDADE_NIVEIS)
    if extras_idade:
        raise ValueError(f"Categorias inesperadas em applicant_age: {sorted(extras_idade)}")

    fluxo = [_registrar_corte(df.iloc[0:0], "Amostra principal (entrada)", len(df))]

    # Renda <= 0
    ok = df["income"] > 0
    fluxo.append(_registrar_corte(df[~ok], "Renda <= 0 (log indefinido)", int(ok.sum())))

    # Winsorização do LTV e transformações
    lo, hi = df.loc[ok, "combined_loan_to_value_ratio"].quantile(WINSOR_LTV)
    df["ltv"] = df["combined_loan_to_value_ratio"].clip(lo, hi)
    df["log_income"] = np.log(df["income"].where(ok, 1.0))
    df["renda_nao_pos"] = (~ok).astype(float)
    fluxo.append(_registrar_corte(
        df.iloc[0:0], f"LTV winsorizado em [{lo:.2f}, {hi:.2f}] (nenhuma linha removida)", int(ok.sum())
    ))

    # Criação das dummies e rótulos legíveis
    rotulos = {
        "mulher": "mulher",
        "log_income": "log(renda)",
        "ltv": "LTV combinado (%)",
        "Intercept": "Intercepto",
        "renda_nao_pos": "Renda <= 0 (indicador)",
    }
    grupos_dummies = {}
    for col, niveis, prefixo in [
        ("dti", DTI_NIVEIS, "dti"),
        ("loan_type", TIPO_NIVEIS, "tipo"),
        ("applicant_age", IDADE_NIVEIS, "idade"),
    ]:
        nomes = []
        for i, nivel in enumerate(niveis[1:], start=1):
            nome = f"{prefixo}_{i}"
            df[nome] = (df[col] == nivel).astype(float)
            rotulos[nome] = f"{col} = {nivel} (ref.: {niveis[0]})"
            nomes.append(nome)
        grupos_dummies[prefixo] = nomes

    controles = ["log_income", "ltv"] + grupos_dummies["dti"] + grupos_dummies["tipo"]
    nomes_idade = grupos_dummies["idade"]

    manter = ["lei", "state_code", "applicant_age", "negado", "mulher",
              "renda_nao_pos"] + controles + nomes_idade
    df = df[manter]

    # Subamostras: S1 (base), principal e S2 (todas)
    base = df[ok].reset_index(drop=True)

    sem_estado = base["state_code"].isna()
    fluxo.append(_registrar_corte(base[sem_estado], "state_code faltante", int((~sem_estado).sum())))
    principal = base[~sem_estado]

    sem_idade = principal["applicant_age"].isna()
    fluxo.append(_registrar_corte(principal[sem_idade], "applicant_age faltante", int((~sem_idade).sum())))
    principal = principal[~sem_idade].drop(columns="applicant_age").reset_index(drop=True)

    base = base.drop(columns="applicant_age")
    todas = df.drop(columns="applicant_age").copy()
    todas["state_code"] = todas["state_code"].fillna("SEM_UF")
    return todas, base, principal, controles, nomes_idade, pd.DataFrame(fluxo), rotulos


def criar_especificacoes(
    todas: pd.DataFrame,
    base: pd.DataFrame,
    principal: pd.DataFrame,
    controles: list[str],
    nomes_idade: list[str],
) -> list[dict]:
    x_ctrl = ["mulher"] + controles
    return [
        {"nome": "(1) Só gênero", "tipo": "principal", "x": ["mulher"], "fe": [], "dados": principal},
        {"nome": "(2) + controles", "tipo": "principal", "x": x_ctrl, "fe": [], "dados": principal},
        {"nome": "(3) + EF estado", "tipo": "principal", "x": x_ctrl, "fe": ["state_code"], "dados": principal},
        {"nome": "(4) + EF instituição", "tipo": "principal", "x": x_ctrl, "fe": ["lei"], "dados": principal},
        {
            "nome": "(5) + EF instituição e estado [H1]",
            "tipo": "principal",
            "x": x_ctrl,
            "fe": ["lei", "state_code"],
            "dados": principal,
        },
        {
            "nome": "S1: (4) sem exigir state_code/idade",
            "tipo": "exploratória",
            "x": x_ctrl,
            "fe": ["lei"],
            "dados": base,
        },
        {
            "nome": "S2: (5) todas as linhas",
            "tipo": "exploratória",
            "x": x_ctrl + ["renda_nao_pos"],
            "fe": ["lei", "state_code"],
            "dados": todas,
        },
        {
            "nome": "E1: (5) + faixa etária",
            "tipo": "exploratória",
            "x": x_ctrl + nomes_idade,
            "fe": ["lei", "state_code"],
            "dados": principal,
        },
    ]


def formula_especificacao(spec: dict) -> str:
    formula = f"negado ~ {' + '.join(spec['x'])}"
    return formula + (f" | {' + '.join(spec['fe'])}" if spec["fe"] else "")


# --------------------------------------------------------------------------- #
# 2. Estimação (Logit + AME + verificação MPL)
# --------------------------------------------------------------------------- #

def estimar_especificacao(spec: dict, rotulos: dict) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    colunas = list(dict.fromkeys(["negado", "lei"] + spec["x"] + spec["fe"]))
    dados = spec["dados"][colunas].reset_index(drop=True)

    fml = formula_especificacao(spec)
    vcov = {"CRV1": "lei"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit = pf.feglm(fml, data=dados, family="logit", vcov=vcov)
        mpl = pf.feols(fml, data=dados, vcov=vcov)

    coefs, erros = logit.coef(), logit.se()
    nomes = list(coefs.index)
    b, ep = float(coefs["mulher"]), float(erros["mulher"])
    b_mpl, ep_mpl = float(mpl.coef()["mulher"]), float(mpl.se()["mulher"])

    # Amostra efetivamente mantida no logit (após singletons e separação)
    info = logit.sample_info
    mantida = np.ones(len(dados), dtype=bool)
    mantida[np.fromiter(info.dropped_row_index, dtype=np.int64)] = False
    est = dados.loc[mantida]

    # Efeito Marginal Médio (AME) via contrafactuais + método delta
    link = np.asarray(logit.predict(type="link"), dtype=float)
    eta0 = link - b * est["mulher"].to_numpy()
    p1, p0 = expit(eta0 + b), expit(eta0)
    ame = float((p1 - p0).mean())

    X = np.column_stack([np.ones(len(est)) if n == "Intercept" else est[n].to_numpy() for n in nomes])
    grad = X.T @ (p1 * (1 - p1) - p0 * (1 - p0)) / len(est)
    grad[nomes.index("mulher")] = (p1 * (1 - p1)).mean()
    ep_ame = float(np.sqrt(grad @ np.asarray(logit._vcov, dtype=float) @ grad))

    cortes = info.dropped_by_stage
    resumo = {
        "especificacao": spec["nome"],
        "tipo": spec["tipo"],
        "n_obs": int(info.n_obs),
        "n_instituicoes": int(est["lei"].nunique()),
        "removidas_singleton": int(getattr(cortes, "singleton", 0)),
        "removidas_separacao": int(getattr(cortes, "separation", 0)),
        "coef_genero": b,
        "ep_cluster": ep,
        "odds_ratio": float(np.exp(b)),
        "or_ic95_inf": float(np.exp(b - Z95 * ep)),
        "or_ic95_sup": float(np.exp(b + Z95 * ep)),
        "p_valor": float(2 * norm.sf(abs(b / ep))),
        "ame_pp": 100 * ame,
        "ame_ic95_inf_pp": 100 * (ame - Z95 * ep_ame),
        "ame_ic95_sup_pp": 100 * (ame + Z95 * ep_ame),
        "ame_ic": "delta (EF tratados como fixos)" if spec["fe"] else "delta",
        "mpl_pp": 100 * b_mpl,
        "mpl_ic95_inf_pp": 100 * (b_mpl - Z95 * ep_mpl),
        "mpl_ic95_sup_pp": 100 * (b_mpl + Z95 * ep_mpl),
        "formula": fml,
        "tempo_s": round(time.time() - t0, 1),
    }

    tab_coefs = pd.DataFrame({
        "especificacao": spec["nome"],
        "termo": [rotulos.get(n, n) for n in nomes],
        "coef": coefs.to_numpy(),
        "ep_cluster": erros.to_numpy(),
        "p_valor": 2 * norm.sf(np.abs(coefs.to_numpy() / erros.to_numpy())),
        "odds_ratio": np.exp(coefs.to_numpy()),
        "or_ic95_inf": np.exp(coefs.to_numpy() - Z95 * erros.to_numpy()),
        "or_ic95_sup": np.exp(coefs.to_numpy() + Z95 * erros.to_numpy()),
    })
    return resumo, tab_coefs


def estimar_especificacoes(
    especificacoes: list[dict], rotulos: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resultados = []
    coeficientes = []
    for spec in especificacoes:
        print(f"\nEstimando {spec['nome']}")
        resumo, tab_coefs = estimar_especificacao(spec, rotulos)
        resultados.append(resumo)
        coeficientes.append(tab_coefs)
        print(
            f"  N = {resumo['n_obs']:,} | OR = {resumo['odds_ratio']:.3f} | "
            f"AME = {resumo['ame_pp']:.2f} p.p. ({resumo['tempo_s']}s)"
        )
    return pd.DataFrame(resultados), pd.concat(coeficientes, ignore_index=True)


# 3. Formatação da tabela Markdown

def gerar_markdown(res: pd.DataFrame) -> str:
    fmt = lambda x, c: f"{x:.{c}f}".replace(".", ",")
    linhas = [
        "| Especificação | Tipo | N | OR do gênero (IC 95%) | p | AME (p.p.) (IC 95%) | Verificação MPL (p.p.) (IC 95%) |",
        "|" + "---|" * 7,
    ]
    for r in res.itertuples():
        p_str = "<0,001" if r.p_valor < 1e-3 else fmt(r.p_valor, 3)
        linhas.append(
            f"| {r.especificacao} | {r.tipo} | {f'{r.n_obs:,}'.replace(',', '.')} "
            f"| {fmt(r.odds_ratio, 3)} ({fmt(r.or_ic95_inf, 3)}; {fmt(r.or_ic95_sup, 3)}) "
            f"| {p_str} "
            f"| {fmt(r.ame_pp, 2)} ({fmt(r.ame_ic95_inf_pp, 2)}; {fmt(r.ame_ic95_sup_pp, 2)}) "
            f"| {fmt(r.mpl_pp, 2)} ({fmt(r.mpl_ic95_inf_pp, 2)}; {fmt(r.mpl_ic95_sup_pp, 2)}) |"
        )
    linhas.append(
        "\nNotas: OR > 1 indica maior chance de negação para mulheres. Erros-padrão clusterizados "
        "por instituição (`lei`). AME calculado por contrafactuais (IC via método delta); MPL traz "
        "o efeito linear em p.p. como verificação. S1, S2 e E1 são análises exploratórias."
    )
    return "\n".join(linhas) + "\n"


def salvar_resultados(
    resultados: pd.DataFrame,
    coeficientes: pd.DataFrame,
    fluxo: pd.DataFrame,
    caminho_dados: Path = ARQUIVO_DADOS,
    amostra: int | None = AMOSTRA_TESTE,
    pasta_saida: Path = PASTA_SAIDA,
) -> None:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(pasta_saida / "etapa_04_h1_especificacoes.csv", index=False)
    (pasta_saida / "etapa_04_h1_especificacoes.md").write_text(
        gerar_markdown(resultados), encoding="utf-8"
    )
    coeficientes.to_csv(pasta_saida / "etapa_04_h1_coeficientes.csv", index=False)
    fluxo.to_csv(pasta_saida / "etapa_04_h1_fluxo_amostra.csv", index=False)

    meta = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "dados": str(caminho_dados),
        "amostra_teste": amostra,
        "semente": SEMENTE,
    }
    (pasta_saida / "etapa_04_h1_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #

def main() -> dict[str, pd.DataFrame | list[dict]]:
    print(f"Carregando {ARQUIVO_DADOS}")

    todas, base, principal, controles, nomes_idade, fluxo_df, rotulos = preparar_dados(
        ARQUIVO_DADOS, AMOSTRA_TESTE
    )
    print("\nFluxo da amostra:\n", fluxo_df.to_string(index=False))

    especificacoes = criar_especificacoes(todas, base, principal, controles, nomes_idade)
    res_df, coefs_df = estimar_especificacoes(especificacoes, rotulos)
    salvar_resultados(res_df, coefs_df, fluxo_df)

    h1 = res_df.loc[res_df["especificacao"].str.contains(r"\[H1\]")].iloc[0]
    decisao = "rejeita H0" if h1["p_valor"] < 0.05 else "não rejeita H0"
    print("\n" + "=" * 65)
    print(f"H1: OR = {h1['odds_ratio']:.3f} (IC 95%: {h1['or_ic95_inf']:.3f}; {h1['or_ic95_sup']:.3f}) | "
          f"AME = {h1['ame_pp']:.2f} p.p. | p = {h1['p_valor']:.4f} -> {decisao}")
    return {
        "resultados": res_df,
        "coeficientes": coefs_df,
        "fluxo": fluxo_df,
        "especificacoes": especificacoes,
    }


if __name__ == "__main__":
    main()