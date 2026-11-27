"""Etapa D (H2): Gradient Boosting com e sem gênero na previsão de negação."""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preparo_hmda import (
    FEATURES_BASE,
    SEED,
    carregar_amostra_modelagem,
    criar_modelo_gradient_boosting,
    dividir_amostra,
    escolher_limiar,
    project_root,
)

# --------------------------------------------------------------------------- #
# Caminhos e parâmetros (Protocolo, seção 3.2)
# --------------------------------------------------------------------------- #

RAIZ = project_root()
ARQUIVO_DADOS = RAIZ / "data" / "amostra_principal.parquet"
PASTA_SAIDA = RAIZ / "resultados"

AMOSTRA_TESTE = None
PROPORCAO_TESTE = 0.30


# 1. Seleção de hiperparâmetros
def buscar_hiperparametros(df_treino: pd.DataFrame) -> GridSearchCV:
    amostra_busca = df_treino.sample(
        n=min(100_000, len(df_treino)), random_state=SEED
    )
    busca = GridSearchCV(
        criar_modelo_gradient_boosting(),
        {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    )
    busca.fit(amostra_busca[FEATURES_BASE], amostra_busca["negado"])
    return busca


# 2. Treinamento, limiar por CV no treino e avaliação no teste

def avaliar_modelo(
    nome: str,
    features: list[str],
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    hiperparametros: dict,
) -> tuple[dict, HistGradientBoostingClassifier, np.ndarray, np.ndarray]:
    t0 = time.time()
    X_treino, y_treino = df_treino[features], df_treino["negado"].to_numpy()
    X_teste, y_teste = df_teste[features], df_teste["negado"].to_numpy()

    modelo_base = criar_modelo_gradient_boosting(**hiperparametros)

    # Escolha do limiar no treino via validação cruzada (taxa prevista == observada)
    taxa_obs_treino = float(y_treino.mean())
    limiar = escolher_limiar(modelo_base, X_treino, y_treino)

    # Ajuste final no treino completo e predição no teste
    modelo = clone(modelo_base).fit(X_treino, y_treino)
    proba_teste = modelo.predict_proba(X_teste)[:, 1]
    pred_teste = (proba_teste >= limiar).astype(int)

    # Prévia das taxas previstas por gênero no teste (detalhada com bootstrap na Etapa E)
    m_teste = df_teste["mulher"].to_numpy() == 1
    taxa_pred_f = float(pred_teste[m_teste].mean())
    taxa_pred_m = float(pred_teste[~m_teste].mean())

    metricas = {
        "modelo": nome,
        "n_features": len(features),
        "n_treino": len(df_treino),
        "n_teste": len(df_teste),
        "auc_roc": float(roc_auc_score(y_teste, proba_teste)),
        "pr_auc": float(average_precision_score(y_teste, proba_teste)),
        "brier_score": float(brier_score_loss(y_teste, proba_teste)),
        "limiar_cv_treino": limiar,
        "taxa_obs_treino": taxa_obs_treino,
        "taxa_obs_teste": float(y_teste.mean()),
        "taxa_pred_teste": float(pred_teste.mean()),
        "taxa_pred_mulher": taxa_pred_f,
        "taxa_pred_homem": taxa_pred_m,
        "dp_previa_pp": 100 * (taxa_pred_f - taxa_pred_m),
        "tempo_s": round(time.time() - t0, 1),
    }
    return metricas, modelo, proba_teste, pred_teste


# 3. Formatação da tabela Markdown

def gerar_markdown(res: pd.DataFrame) -> str:
    fmt = lambda x, c: f"{x:.{c}f}".replace(".", ",")
    linhas = [
        "| Modelo | AUC-ROC | PR-AUC | Brier Score | Limiar (CV treino) | Taxa prev. teste | DP prévia (p.p.) |",
        "|" + "---|" * 7,
    ]
    for r in res.itertuples():
        linhas.append(
            f"| {r.modelo} | {fmt(r.auc_roc, 4)} | {fmt(r.pr_auc, 4)} | {fmt(r.brier_score, 4)} "
            f"| {fmt(r.limiar_cv_treino, 4)} | {fmt(100 * r.taxa_pred_teste, 2)}% "
            f"| {fmt(r.dp_previa_pp, 2)} |"
        )
    linhas.append(
        "\nNotas: Divisão 70/30 estratificada conjuntamente por `negado` e `mulher` (`random_state=444`). "
        "Limiar escolhido via CV 5-fold no treino para igualar a taxa prevista à observada. "
        "DP prévia = P(ŷ=1 | Mulher) − P(ŷ=1 | Homem) no teste (IC por bootstrap na Etapa E)."
    )
    return "\n".join(linhas) + "\n"


def executar_modelagem(
    caminho_dados: Path = ARQUIVO_DADOS,
    amostra: int | None = AMOSTRA_TESTE,
    pasta_saida: Path = PASTA_SAIDA,
) -> dict:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    print(f"Carregando {caminho_dados}")
    df = carregar_amostra_modelagem(caminho=caminho_dados, amostra=amostra)

    # Divisão 70/30 idêntica, estratificada pelo alvo e pelo gênero
    df_treino, df_teste, idx_treino, idx_teste = dividir_amostra(
        df, test_size=PROPORCAO_TESTE
    )
    print(f"Base: {len(df):,} linhas | Treino: {len(df_treino):,} | Teste: {len(df_teste):,}")
    busca = buscar_hiperparametros(df_treino)
    hiperparametros = busca.best_params_
    print(f"Melhores hiperparâmetros: {hiperparametros}")

    especs = [
        ("Com gênero", ["mulher"] + FEATURES_BASE),
        ("Sem gênero", FEATURES_BASE),
    ]

    resultados = []
    modelos_salvos = {}
    predicoes_teste = df_teste[["negado", "mulher"] + FEATURES_BASE].copy()

    for nome, feats in especs:
        print(f"\nTreinando Gradient Boosting: {nome}")
        metricas, modelo, proba, pred = avaliar_modelo(
            nome, feats, df_treino, df_teste, hiperparametros
        )
        resultados.append(metricas)
        modelos_salvos[nome] = {
            "modelo": modelo,
            "features": feats,
            "limiar": metricas["limiar_cv_treino"],
        }

        sufixo = "com_genero" if nome == "Com gênero" else "sem_genero"
        predicoes_teste[f"proba_{sufixo}"] = proba
        predicoes_teste[f"pred_{sufixo}"] = pred

        print(
            f"  AUC-ROC = {metricas['auc_roc']:.4f} | PR-AUC = {metricas['pr_auc']:.4f} | "
            f"Brier = {metricas['brier_score']:.4f} | DP prévia = {metricas['dp_previa_pp']:.2f} p.p. "
            f"({metricas['tempo_s']}s)"
        )

    res_df = pd.DataFrame(resultados)
    res_df.to_csv(pasta_saida / "etapa_05_h2_performance.csv", index=False)
    (pasta_saida / "etapa_05_h2_performance.md").write_text(gerar_markdown(res_df), encoding="utf-8")

    # Artefatos salvos para as Etapas E (Fairness) e F (Proxies/SHAP)
    predicoes_teste.to_parquet(pasta_saida / "etapa_05_h2_predicoes_teste.parquet", index=False)
    joblib.dump(
        {
            "formato": 1,
            "modelos": modelos_salvos,
            "idx_treino": idx_treino,
            "idx_teste": idx_teste,
            "hiperparametros": hiperparametros,
        },
        pasta_saida / "etapa_05_h2_modelos.joblib",
    )

    meta = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "dados": str(caminho_dados),
        "amostra_teste": amostra,
        "semente": SEED,
        "proporcao_teste": PROPORCAO_TESTE,
        "features_base": FEATURES_BASE,
    }
    (pasta_saida / "etapa_05_h2_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados salvos em {pasta_saida}")
    return {
        "dados": df,
        "treino": df_treino,
        "teste": df_teste,
        "idx_treino": idx_treino,
        "idx_teste": idx_teste,
        "hiperparametros": hiperparametros,
        "busca": busca,
        "metricas": res_df,
        "modelos": {nome: item["modelo"] for nome, item in modelos_salvos.items()},
        "limiares": {nome: item["limiar"] for nome, item in modelos_salvos.items()},
        "probabilidades": {
            nome: predicoes_teste[f"proba_{'com_genero' if nome == 'Com gênero' else 'sem_genero'}"].to_numpy()
            for nome, _ in especs
        },
        "predicoes": {
            nome: predicoes_teste[f"pred_{'com_genero' if nome == 'Com gênero' else 'sem_genero'}"].to_numpy()
            for nome, _ in especs
        },
    }


def main() -> None:
    executar_modelagem()


if __name__ == "__main__":
    main()
