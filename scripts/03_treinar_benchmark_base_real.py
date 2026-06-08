"""
FireGuard Orbit - Treinar benchmark com a base real consolidada.

Entrada:
  data/dados_fireguard_orbit_real.csv

Saídas:
  benchmark_modelos_real.csv
  resultados_modelo_real.json
"""

from pathlib import Path
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import inspect
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dados_fireguard_orbit_real.csv"
RANDOM_STATE = 42


def make_ohe():
    """Compatível com versões antigas e novas do scikit-learn."""
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def score_roc(pipe, X_test, y_test):
    if hasattr(pipe, "predict_proba"):
        try:
            return roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        return roc_auc_score(y_test, pipe.decision_function(X_test))
    return None


def main():
    df = pd.read_csv(DATASET)

    alvo = "risco_alto_queimada"

    features_categoricas = ["regiao", "bioma"]
    features_numericas = [
        "mes", "dia_do_ano",
        "temperatura_c", "temperatura_max_c", "temperatura_min_c",
        "umidade_pct", "precipitacao_mm", "vento_kmh",
        "dias_sem_chuva", "chuva_acumulada_7d", "umidade_media_7d", "temperatura_media_7d",
        "focos_dia_anterior", "focos_acumulados_7d", "media_focos_7d", "media_focos_15d",
    ]

    X = df[features_categoricas + features_numericas]
    y = df[alvo]

    modelos = {
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    resultados = []
    pipelines = {}
    for nome, modelo in modelos.items():
        print(f"Treinando {nome}...")
        pre = ColumnTransformer([
            ("cat", make_ohe(), features_categoricas),
            ("num", StandardScaler(), features_numericas),
        ])
        pipe = Pipeline([("pre", pre), ("modelo", modelo)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        auc = score_roc(pipe, X_test, y_test)

        resultados.append({
            "modelo": nome,
            "accuracy": accuracy_score(y_test, pred),
            "precision_risco_alto": precision_score(y_test, pred, zero_division=0),
            "recall_risco_alto": recall_score(y_test, pred, zero_division=0),
            "f1_risco_alto": f1_score(y_test, pred, zero_division=0),
            "roc_auc": auc,
        })
        pipelines[nome] = pipe

    bench = pd.DataFrame(resultados).sort_values(
        by=["recall_risco_alto", "f1_risco_alto", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    bench.to_csv(ROOT / "benchmark_modelos_real.csv", index=False, encoding="utf-8")

    melhor_nome = bench.iloc[0]["modelo"]
    melhor = pipelines[melhor_nome]
    pred = melhor.predict(X_test)
    auc_final = score_roc(melhor, X_test, y_test)

    saida_json = {
        "modelo_escolhido": melhor_nome,
        "criterio_de_escolha": "Maior recall da classe risco_alto_queimada; desempate por F1 e ROC-AUC.",
        "base_utilizada": "data/dados_fireguard_orbit_real.csv",
        "features_categoricas": features_categoricas,
        "features_numericas": features_numericas,
        "metricas": bench.iloc[0].to_dict(),
        "matriz_confusao": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "roc_auc_calculado_no_teste": auc_final,
    }

    with open(ROOT / "resultados_modelo_real.json", "w", encoding="utf-8") as f:
        json.dump(saida_json, f, ensure_ascii=False, indent=2)

    print("\nBenchmark salvo em benchmark_modelos_real.csv")
    print("Resultados salvos em resultados_modelo_real.json")
    print("\nRanking:")
    print(bench)


if __name__ == "__main__":
    main()
