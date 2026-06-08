"""
GS - Cognitive Computing, Computer Vision and IoT Systems | FIAP
Projeto: FireGuard Orbit

Versão final: focos reais do INPE + dados meteorológicos reais da NASA POWER.
"""

from pathlib import Path
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
import inspect

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "data" / "dados_fireguard_orbit_real.csv"
GRAF_DIR = BASE_DIR / "graficos"
GRAF_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "risco_alto_queimada"
CAT_COLS = ["regiao", "bioma"]
NUM_COLS = [
    "mes", "dia_do_ano",
    "temperatura_c", "temperatura_max_c", "temperatura_min_c",
    "umidade_pct", "precipitacao_mm", "vento_kmh",
    "dias_sem_chuva", "chuva_acumulada_7d", "umidade_media_7d", "temperatura_media_7d",
    "focos_dia_anterior", "focos_acumulados_7d", "media_focos_7d", "media_focos_15d",
]
FEATURES = CAT_COLS + NUM_COLS


def make_ohe():
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def carregar_dados():
    if not DATASET.exists():
        raise FileNotFoundError("Base real não encontrada. Rode scripts/02_montar_base_real_fireguard.py")
    df = pd.read_csv(DATASET, parse_dates=["data"])
    faltantes = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltantes:
        raise KeyError(f"Colunas faltantes na base real: {faltantes}")
    return df


def preprocessor():
    return ColumnTransformer([
        ("cat", make_ohe(), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])


def modelos():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=100),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }


def main():
    print("=" * 70)
    print("FireGuard Orbit — base final real")
    print("INPE Queimadas + NASA POWER + variáveis históricas")
    print("=" * 70)

    df = carregar_dados()
    print(f"Linhas: {len(df):,} | Colunas: {df.shape[1]}")
    print(f"Período: {df['data'].min().date()} a {df['data'].max().date()}")
    print(f"Classe risco alto: {df[TARGET].mean():.1%}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    resultados = []
    pipes = {}
    for nome, modelo in modelos().items():
        print(f"Treinando {nome}...")
        pipe = Pipeline([("pre", preprocessor()), ("modelo", modelo)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        resultados.append({
            "modelo": nome,
            "accuracy": accuracy_score(y_test, pred),
            "precision_risco_alto": precision_score(y_test, pred, zero_division=0),
            "recall_risco_alto": recall_score(y_test, pred, zero_division=0),
            "f1_risco_alto": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        pipes[nome] = pipe

    bench = pd.DataFrame(resultados).sort_values(
        ["recall_risco_alto", "f1_risco_alto", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    bench.to_csv(BASE_DIR / "benchmark_modelos_real.csv", index=False, encoding="utf-8")

    melhor_nome = bench.iloc[0]["modelo"]
    melhor = pipes[melhor_nome]
    pred = melhor.predict(X_test)
    proba = melhor.predict_proba(X_test)[:, 1]

    saida = {
        "modelo_escolhido": melhor_nome,
        "criterio_de_escolha": "Maior recall da classe risco_alto_queimada; desempate por F1 e ROC-AUC.",
        "base_utilizada": "data/dados_fireguard_orbit_real.csv",
        "features_categoricas": CAT_COLS,
        "features_numericas": NUM_COLS,
        "metricas": bench.iloc[0].to_dict(),
        "matriz_confusao": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "roc_auc_calculado_no_teste": roc_auc_score(y_test, proba),
    }
    with open(BASE_DIR / "resultados_modelo_real.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Risco Alto"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=True)
    ax.set_title(f"Matriz de Confusão — {melhor_nome}\nBase real INPE + NASA POWER")
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "04_matriz_confusao.png", dpi=160)
    plt.close()

    print("\nRanking:")
    print(bench)
    print("\nModelo escolhido:", melhor_nome)
    print(classification_report(y_test, pred, target_names=["Normal", "Risco Alto"]))


if __name__ == "__main__":
    main()
