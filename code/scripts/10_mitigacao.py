#!/usr/bin/env python3
"""Etapa H: avalia reponderação e limiares por grupo."""

from __future__ import annotations

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from preparo_hmda import FEATURES_BASE, carregar_amostra_modelagem, project_root

RAIZ = project_root()
ARQUIVO_DADOS = RAIZ / "data" / "amostra_principal.parquet"
ARQUIVO_MODELOS = RAIZ / "resultados" / "etapa_05_h2_modelos.joblib"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_FIGURAS = RAIZ / "figuras"
FEATURES_POR_MODELO = {
    "Com gênero": ["mulher"] + FEATURES_BASE,
    "Sem gênero": FEATURES_BASE,
}


def carregar_base_principal() -> pd.DataFrame:
    """Carrega a amostra usada para treinar os modelos da Etapa D."""
    return carregar_amostra_modelagem(caminho=ARQUIVO_DADOS)


def pesos_reweighing(genero: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula os pesos de reponderação usando apenas o treino."""
    tab = pd.crosstab(pd.Series(genero, name="g"), pd.Series(y, name="y"), normalize=True)
    pg = pd.Series(genero).value_counts(normalize=True)
    py = pd.Series(y).value_counts(normalize=True)
    pesos = np.empty(len(y), dtype=float)
    for g in np.unique(genero):
        for classe in np.unique(y):
            conjunta = float(tab.loc[g, classe])
            if conjunta == 0:
                raise ValueError(f"Célula gênero={g}, negado={classe} sem observações no treino")
            pesos[(genero == g) & (y == classe)] = float(pg[g] * py[classe] / conjunta)
    return pesos


def metricas(y: np.ndarray, proba: np.ndarray, pred: np.ndarray, genero: np.ndarray) -> dict:
    mulher, homem = genero == 1, genero == 0
    neg_f = float(pred[mulher].mean())
    neg_m = float(pred[homem].mean())
    aprov_f, aprov_m = 1.0 - neg_f, 1.0 - neg_m
    di = aprov_f / aprov_m if aprov_m else np.nan
    return {
        "acuracia": float(accuracy_score(y, pred)),
        "auc_roc": float(roc_auc_score(y, proba)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "taxa_negacao_mulher": neg_f,
        "taxa_negacao_homem": neg_m,
        "dp": neg_f - neg_m,
        "disparate_impact": di,
    }


def escolher_limites(y: np.ndarray, proba: np.ndarray, genero: np.ndarray) -> tuple[float, float]:
    """Escolhe no teste os limiares que maximizam a acurácia com DI >= 0,80."""
    candidatos = np.unique(np.quantile(proba, np.linspace(0, 1, 201)))
    # Inclui os limites extremos da grade.
    candidatos = np.unique(np.r_[0.0, candidatos, 1.0 + 1e-12])
    mulher, homem = genero == 1, genero == 0
    def sumarizar(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        taxas, corretos = [], []
        for limiar in candidatos:
            pred = proba[mask] >= limiar
            taxas.append(float(pred.mean()))
            corretos.append(int((pred == y[mask]).sum()))
        return np.asarray(taxas), np.asarray(corretos), candidatos

    txf, corretos_f, _ = sumarizar(mulher)
    txm, corretos_m, _ = sumarizar(homem)
    aprov_f, aprov_m = 1 - txf[:, None], 1 - txm[None, :]
    di = np.divide(aprov_f, aprov_m, out=np.full((len(txf), len(txm)), np.inf), where=aprov_m != 0)
    acertos = corretos_f[:, None] + corretos_m[None, :]
    dp_abs = np.abs(txf[:, None] - txm[None, :])
    distancia = np.abs(candidatos[:, None] - candidatos[None, :])
    # Desempate: menor |DP| e limiares mais próximos.
    elegivel = di >= 0.80
    if elegivel.any():
        acertos_obj = np.where(elegivel, acertos, -1)
        max_acertos = acertos_obj.max()
        mascarar = elegivel & (acertos == max_acertos)
    else:
        menor_distancia = np.nanmin(np.abs(di - 0.80))
        mascarar = np.abs(di - 0.80) == menor_distancia
        max_acertos = np.where(mascarar, acertos, -1).max()
        mascarar &= acertos == max_acertos
    dp_obj = np.where(mascarar, dp_abs, np.inf)
    min_dp = dp_obj.min()
    mascarar &= dp_abs == min_dp
    dist_obj = np.where(mascarar, distancia, np.inf)
    i, j = np.unravel_index(np.argmin(dist_obj), dist_obj.shape)
    melhor = (None, float(candidatos[i]), float(candidatos[j]))
    return melhor[1], melhor[2]


def aplicar_limites(proba: np.ndarray, genero: np.ndarray, tf: float, tm: float) -> np.ndarray:
    return (proba >= np.where(genero == 1, tf, tm)).astype(int)


def tradeoff_limites(
    y: np.ndarray, proba: np.ndarray, genero: np.ndarray, modelo: str, estrategia: str
) -> pd.DataFrame:
    candidatos = np.unique(np.quantile(proba, np.linspace(0, 1, 101)))
    mulher, homem = genero == 1, genero == 0
    n_f, n_m = int(mulher.sum()), int(homem.sum())
    tf_rates = np.array([(proba[mulher] >= t).mean() for t in candidatos])
    tm_rates = np.array([(proba[homem] >= t).mean() for t in candidatos])
    tf_correct = np.array([((proba[mulher] >= t) == y[mulher]).sum() for t in candidatos])
    tm_correct = np.array([((proba[homem] >= t) == y[homem]).sum() for t in candidatos])
    neg_f, neg_m = tf_rates[:, None], tm_rates[None, :]
    aprov_f, aprov_m = 1 - neg_f, 1 - neg_m
    di = np.divide(aprov_f, aprov_m, out=np.full((len(candidatos), len(candidatos)), np.inf), where=aprov_m != 0)
    dp = neg_f - neg_m
    acc = (tf_correct[:, None] + tm_correct[None, :]) / len(y)
    auc = float(roc_auc_score(y, proba))
    return pd.DataFrame({
        "modelo": modelo, "estrategia": estrategia,
        "limiar_mulher": np.repeat(candidatos, len(candidatos)),
        "limiar_homem": np.tile(candidatos, len(candidatos)),
        "acuracia": acc.ravel(), "auc_roc": auc,
        "dp": dp.ravel(), "disparate_impact": di.ravel(),
        "taxa_negacao_mulher": np.repeat(tf_rates, len(candidatos)),
        "taxa_negacao_homem": np.tile(tm_rates, len(candidatos)),
    })


def construir_grafico(frente: pd.DataFrame, tabela: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    cores = {"Com gênero": "#245a81", "Sem gênero": "#c65d35"}
    for (nome, estrategia), sub in frente.groupby(["modelo", "estrategia"]):
        cor = cores[nome]
        marcador = "." if estrategia == "Ajuste de Limiar" else "x"
        axs[0, 0].scatter(sub["disparate_impact"], sub["acuracia"], s=8, alpha=.08, color=cor, marker=marcador)
        axs[0, 1].scatter(sub["dp"].abs(), sub["acuracia"], s=8, alpha=.08, color=cor, marker=marcador)
        axs[1, 0].scatter(sub["disparate_impact"], sub["auc_roc"], s=8, alpha=.08, color=cor, marker=marcador)
        axs[1, 1].scatter(sub["dp"].abs(), sub["auc_roc"], s=8, alpha=.08, color=cor, marker=marcador)
    for _, r in tabela.iterrows():
        cor = cores[r["modelo"]]
        marker = {"Baseline": "o", "Reweighing": "s", "Ajuste de Limiar": "^", "Combinado": "D"}[r["estrategia"]]
        for ax, xcol, ycol in [
            (axs[0, 0], "disparate_impact", "acuracia"), (axs[0, 1], "dp_abs", "acuracia"),
            (axs[1, 0], "disparate_impact", "auc_roc"), (axs[1, 1], "dp_abs", "auc_roc"),
        ]:
            ax.scatter(r[xcol], r[ycol], color=cor, marker=marker, s=48, edgecolor="black", linewidth=.35)
    titulos = ["Acurácia vs. Disparate Impact", "Acurácia vs. |DP|", "AUC-ROC vs. Disparate Impact", "AUC-ROC vs. |DP|"]
    for ax, titulo in zip(axs.flat, titulos):
        ax.set_title(titulo)
        ax.grid(alpha=.25)
    for ax in (axs[0, 0], axs[1, 0]):
        ax.axvline(.8, color="#a31d2d", linestyle="--", linewidth=1.3, label="Regra dos 80%")
        # Razões extremas surgem quando quase ninguém de um grupo é aprovado;
        # manter o eixo limitado preserva a leitura da região operacional.
        ax.set_xlim(0, 2)
        ax.legend(fontsize=8)
        ax.set_xlabel("Disparate Impact (aprovação mulher / homem)")
    for ax in (axs[0, 1], axs[1, 1]):
        ax.set_xlabel("|Diferença de Paridade Demográfica|")
    axs[0, 0].set_ylabel("Acurácia")
    axs[0, 1].set_ylabel("Acurácia")
    axs[1, 0].set_ylabel("AUC-ROC")
    axs[1, 1].set_ylabel("AUC-ROC")
    fig.suptitle("Etapa H — Desempenho e disparidade\nPontos: estratégias; nuvens: limiares por grupo", fontsize=14)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "etapa_10_mitigacao_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    df = carregar_base_principal()
    artefatos = joblib.load(ARQUIVO_MODELOS)
    if artefatos.get("formato") != 1:
        raise ValueError("Formato do modelo incompatível. Execute a Etapa D novamente.")
    idx_treino = np.asarray(artefatos["idx_treino"], dtype=np.int64)
    idx_teste = np.asarray(artefatos["idx_teste"], dtype=np.int64)
    indices = np.concatenate([idx_treino, idx_teste])
    if len(df) != len(indices) or not np.array_equal(np.sort(indices), np.arange(len(df))):
        raise ValueError(
            f"A amostra tem {len(df):,} linhas, mas o modelo registra {len(indices):,}. "
            "Execute a Etapa D novamente para salvar um split compatível."
        )
    treino, teste = df.iloc[idx_treino].reset_index(drop=True), df.iloc[idx_teste].reset_index(drop=True)
    y_treino, y_teste = treino["negado"].to_numpy(), teste["negado"].to_numpy()
    genero_treino, genero_teste = treino["mulher"].to_numpy(), teste["mulher"].to_numpy()
    print(f"Base: {len(df):,} | Treino: {len(treino):,} | Teste: {len(teste):,}")

    linhas, frentes = [], []
    for nome, features in FEATURES_POR_MODELO.items():
        salvo = artefatos["modelos"][nome]
        if list(salvo["features"]) != features or not set(features).issubset(df.columns):
            raise ValueError(f"As variáveis do modelo não correspondem às da Etapa D: {nome}")
        modelo_base = clone(salvo["modelo"])
        proba_base = salvo["modelo"].predict_proba(teste[features])[:, 1]
        y_limiar = (proba_base >= float(salvo["limiar"])).astype(int)
        # Os pesos são calculados apenas no treino.
        pesos = pesos_reweighing(genero_treino, y_treino)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo_rw = clone(modelo_base).fit(treino[features], y_treino, sample_weight=pesos)
        proba_rw = modelo_rw.predict_proba(teste[features])[:, 1]

        limite_f, limite_m = escolher_limites(y_teste, proba_base, genero_teste)
        limite_rw_f, limite_rw_m = escolher_limites(y_teste, proba_rw, genero_teste)
        previsoes = {
            "Baseline": (proba_base, y_limiar, float(salvo["limiar"]), float(salvo["limiar"])),
            "Reweighing": (proba_rw, (proba_rw >= float(salvo["limiar"])).astype(int), float(salvo["limiar"]), float(salvo["limiar"])),
            "Ajuste de Limiar": (proba_base, aplicar_limites(proba_base, genero_teste, limite_f, limite_m), limite_f, limite_m),
            "Combinado": (proba_rw, aplicar_limites(proba_rw, genero_teste, limite_rw_f, limite_rw_m), limite_rw_f, limite_rw_m),
        }
        base_m = metricas(y_teste, *previsoes["Baseline"][:2], genero_teste)
        for estrategia, (proba, pred, tf, tm) in previsoes.items():
            m = metricas(y_teste, proba, pred, genero_teste)
            linhas.append({
                "modelo": nome, "estrategia": estrategia, "limiar_mulher": tf, "limiar_homem": tm,
                **m,
                "dp_abs": abs(m["dp"]),
                "queda_disparidade_pct": (100 * (1 - abs(m["dp"]) / abs(base_m["dp"]))) if base_m["dp"] else np.nan,
                "queda_gap_di_080_pct": (100 * (1 - abs(m["disparate_impact"] - 1) / abs(base_m["disparate_impact"] - 1))) if abs(base_m["disparate_impact"] - 1) > 0 else np.nan,
                "perda_acuracia_pp": 100 * (base_m["acuracia"] - m["acuracia"]),
                "perda_auc_pp": 100 * (base_m["auc_roc"] - m["auc_roc"]),
                "pesos_min": float(pesos.min()), "pesos_max": float(pesos.max()),
            })
        frentes.append(tradeoff_limites(y_teste, proba_base, genero_teste, nome, "Ajuste de Limiar"))
        frentes.append(tradeoff_limites(y_teste, proba_rw, genero_teste, nome, "Combinado"))
        print(f"{nome}: AUC={base_m['auc_roc']:.4f}, DI={base_m['disparate_impact']:.4f}; reponderação concluída")

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(PASTA_RESULTADOS / "etapa_10_mitigacao_metricas.csv", index=False)
    frente = pd.concat(frentes, ignore_index=True)
    construir_grafico(frente, resultado)
    gerado = pd.Timestamp.now(tz="America/Sao_Paulo").strftime("%Y-%m-%d %H:%M:%S %Z")
    resumo = [
        "# Etapa H — Mitigação de viés e trade-off",
        "",
        f"Execução: {gerado}. Foram reutilizados os índices do split salvo em `etapa_05_h2_modelos.joblib` "
        f"(treino n={len(treino):,}; teste n={len(teste):,}), com os modelos e hiperparâmetros da Etapa D.",
        "",
        "As métricas descrevem as previsões de negação. O HMDA não informa inadimplência, então não avaliamos erros sobre o risco real.",
        "",
        "A reponderação usa `mulher` e `negado` apenas no treino. Os limiares por grupo foram escolhidos no teste para maximizar a acurácia entre combinações com DI ≥ 0,80. Como o teste participou dessa escolha, os resultados são exploratórios e não medem desempenho fora da amostra.",
        "",
        "## Comparação",
        "",
        "| Modelo | Estratégia | Acurácia | AUC-ROC | F1 | Log-Loss | Negação mulher | Negação homem | ΔDP | DI | Queda | Perda acurácia (p.p.) | Perda AUC (p.p.) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda x: "—" if pd.isna(x) else f"{x:.4f}"
    for r in resultado.itertuples():
        resumo.append(
            f"| {r.modelo} | {r.estrategia} | {fmt(r.acuracia)} | {fmt(r.auc_roc)} | {fmt(r.f1_score)} | {fmt(r.log_loss)} "
            f"| {fmt(r.taxa_negacao_mulher)} | {fmt(r.taxa_negacao_homem)} | {fmt(r.dp)} | {fmt(r.disparate_impact)} "
            f"| {fmt(r.queda_disparidade_pct)}% | {fmt(r.perda_acuracia_pp)} | {fmt(r.perda_auc_pp)} |"
        )
    resumo += [
        "",
        "`ΔDP = P(negação prevista | mulher) − P(negação prevista | homem)`. `DI = P(aprovação prevista | mulher) / P(aprovação prevista | homem)`; referência da regra dos 80%: DI ≥ 0,80. Queda de disparidade é a redução percentual de |ΔDP| frente ao baseline; quedas negativas indicam aumento. Perdas são baseline menos estratégia, em pontos percentuais.",
        "",
        "O gráfico mostra as combinações de limiares avaliadas para os modelos baseline e reponderado. Consulte `etapa_10_mitigacao_metricas.csv` para os limiares e as métricas completas.",
        "",
    ]
    (PASTA_RESULTADOS / "etapa_10_mitigacao_resumo.md").write_text("\n".join(resumo), encoding="utf-8")
    print("Arquivos gerados: resultados/etapa_10_mitigacao_metricas.csv, resultados/etapa_10_mitigacao_resumo.md, figuras/etapa_10_mitigacao_tradeoff.png")


if __name__ == "__main__":
    main()

