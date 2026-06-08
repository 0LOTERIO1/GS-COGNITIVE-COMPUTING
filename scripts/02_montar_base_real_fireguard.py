"""
FireGuard Orbit - Montar base real consolidada.
VERSÃO CORRIGIDA: ajusta o cálculo das janelas móveis para evitar o erro
"Too many levels: Index has only 1 level, not 2" em algumas versões do pandas.

Entrada esperada:
  data/focos_inpe_agregado_2025.csv
  data/nasa_power/nasa_power_biomas_2025.csv

Saída:
  data/dados_fireguard_orbit_real.csv

O que este script faz:
  - Mantém os focos de queimadas da base do projeto.
  - Remove variáveis climáticas antigas/simuladas.
  - Adiciona clima real da NASA POWER por data e bioma.
  - Cria variáveis históricas reais dos focos.
  - Recria a variável alvo como risco alto no próximo dia.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE_FOCOS = DATA / "focos_inpe_agregado_2025.csv"
BASE_CLIMA = DATA / "nasa_power" / "nasa_power_biomas_2025.csv"
SAIDA = DATA / "dados_fireguard_orbit_real.csv"

COLS_REAIS_FOCOS = ["data", "ano", "mes", "regiao", "bioma", "focos_observados"]
CLIMA_COLS = [
    "temperatura_c", "temperatura_max_c", "temperatura_min_c",
    "umidade_pct", "precipitacao_mm", "vento_ms", "vento_kmh"
]


def normalizar_bioma(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def contar_dias_sem_chuva(prec: pd.Series, limite_chuva: float = 1.0) -> pd.Series:
    """Conta dias consecutivos com precipitação abaixo do limite."""
    valores = pd.to_numeric(prec, errors="coerce").fillna(0)
    contador = []
    atual = 0

    for valor in valores:
        if valor < limite_chuva:
            atual += 1
        else:
            atual = 0
        contador.append(atual)

    return pd.Series(contador, index=prec.index)


def validar_colunas(df: pd.DataFrame, colunas: list[str], nome_base: str) -> None:
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise KeyError(f"A base {nome_base} está sem as colunas: {faltantes}")


def preencher_clima_faltante(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche falhas de clima por média do bioma; se ainda faltar, usa média geral."""
    for col in CLIMA_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("bioma")[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    return df


def main():
    if not BASE_FOCOS.exists():
        raise FileNotFoundError(f"Não encontrei {BASE_FOCOS}")

    if not BASE_CLIMA.exists():
        raise FileNotFoundError(
            f"Não encontrei {BASE_CLIMA}. Rode primeiro: python scripts/01_baixar_nasa_power_biomas.py"
        )

    focos = pd.read_csv(BASE_FOCOS)
    clima = pd.read_csv(BASE_CLIMA)

    validar_colunas(focos, COLS_REAIS_FOCOS, "focos_inpe_agregado_2025.csv")
    validar_colunas(clima, ["data", "bioma"] + CLIMA_COLS, "nasa_power_biomas_2025.csv")

    focos["data"] = pd.to_datetime(focos["data"], errors="coerce")
    clima["data"] = pd.to_datetime(clima["data"], errors="coerce")

    focos = focos.dropna(subset=["data"])
    clima = clima.dropna(subset=["data"])

    focos["bioma"] = normalizar_bioma(focos["bioma"])
    clima["bioma"] = normalizar_bioma(clima["bioma"])

    # Mantém apenas colunas confiáveis da base antiga.
    focos = focos[COLS_REAIS_FOCOS].copy()
    focos["focos_observados"] = pd.to_numeric(focos["focos_observados"], errors="coerce").fillna(0)

    # Adiciona clima real por data + bioma.
    df = focos.merge(clima, on=["data", "bioma"], how="left")
    df = preencher_clima_faltante(df)

    df = df.sort_values(["regiao", "bioma", "data"]).reset_index(drop=True)

    # Features históricas de focos.
    # IMPORTANTE: usamos transform(lambda...) para preservar o índice original.
    # Isso evita o erro de reset_index(level=[0, 1]) em versões diferentes do pandas.
    grupo_regiao_bioma = df.groupby(["regiao", "bioma"], group_keys=False)

    df["focos_dia_anterior"] = grupo_regiao_bioma["focos_observados"].transform(
        lambda s: s.shift(1)
    ).fillna(0)

    df["focos_acumulados_7d"] = grupo_regiao_bioma["focos_observados"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0)

    df["media_focos_7d"] = grupo_regiao_bioma["focos_observados"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).fillna(0)

    df["media_focos_15d"] = grupo_regiao_bioma["focos_observados"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=1).mean()
    ).fillna(0)

    # Features climáticas históricas por bioma.
    grupo_bioma = df.groupby("bioma", group_keys=False)

    df["chuva_acumulada_7d"] = grupo_bioma["precipitacao_mm"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0)

    df["umidade_media_7d"] = grupo_bioma["umidade_pct"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).fillna(df["umidade_pct"])

    df["temperatura_media_7d"] = grupo_bioma["temperatura_c"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).fillna(df["temperatura_c"])

    df["dias_sem_chuva"] = grupo_bioma["precipitacao_mm"].transform(
        lambda s: contar_dias_sem_chuva(s)
    ).fillna(0)

    # Alvo: risco alto de queimadas no próximo registro temporal da mesma região/bioma.
    # Usamos shift(-1), então o modelo tenta prever o próximo dia/período com base no histórico anterior.
    df["focos_proximo_dia"] = grupo_regiao_bioma["focos_observados"].transform(lambda s: s.shift(-1))

    # Remove as últimas linhas de cada grupo, pois elas não têm próximo dia conhecido.
    df = df.dropna(subset=["focos_proximo_dia"]).copy()

    # Limiar de risco alto por bioma: top 25% dos próximos dias com mais focos.
    limiares = df.groupby("bioma")["focos_proximo_dia"].quantile(0.75).to_dict()
    df["limiar_risco_bioma"] = df["bioma"].map(limiares)
    df["risco_alto_queimada"] = (df["focos_proximo_dia"] >= df["limiar_risco_bioma"]).astype(int)

    # Campos de data úteis para o modelo e gráficos.
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia_do_ano"] = df["data"].dt.dayofyear

    df = df.reset_index(drop=True)
    df.to_csv(SAIDA, index=False, encoding="utf-8")

    print(f"Base real salva em: {SAIDA}")
    print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
    print("Distribuição do alvo:")
    print(df["risco_alto_queimada"].value_counts(normalize=True).rename("proporcao"))
    print("\nAmostra:")
    print(df.head())


if __name__ == "__main__":
    main()
