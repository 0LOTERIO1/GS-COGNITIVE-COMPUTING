"""
FireGuard Orbit - Baixar dados meteorológicos reais da NASA POWER por bioma.
VERSÃO CORRIGIDA: usa formato JSON em vez de CSV para evitar erro de colunas YEAR/MO/DY.

Como usar:
  1) Substitua o arquivo scripts/01_baixar_nasa_power_biomas.py por este conteúdo.
  2) Rode: python scripts/01_baixar_nasa_power_biomas.py

Saída:
  data/nasa_power/nasa_power_biomas_2025.csv
"""

from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "nasa_power"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ajuste o período conforme o recorte do projeto.
START = "20250101"
END = "20250630"

# Parâmetros climáticos reais usados no projeto.
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS10M"]
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PONTOS_BIOMAS = [
    {"bioma": "Amazônia", "cidade_ref": "Manaus", "lat": -3.1190, "lon": -60.0217},
    {"bioma": "Cerrado", "cidade_ref": "Brasília", "lat": -15.7942, "lon": -47.8825},
    {"bioma": "Pantanal", "cidade_ref": "Corumbá", "lat": -19.0090, "lon": -57.6530},
    {"bioma": "Caatinga", "cidade_ref": "Petrolina", "lat": -9.3891, "lon": -40.5027},
    {"bioma": "Mata Atlântica", "cidade_ref": "São Paulo", "lat": -23.5505, "lon": -46.6333},
    {"bioma": "Pampa", "cidade_ref": "Porto Alegre", "lat": -30.0346, "lon": -51.2177},
]

RENOMEAR = {
    "T2M": "temperatura_c",
    "T2M_MAX": "temperatura_max_c",
    "T2M_MIN": "temperatura_min_c",
    "RH2M": "umidade_pct",
    "PRECTOTCORR": "precipitacao_mm",
    "WS10M": "vento_ms",
}


def baixar_ponto(ponto: dict) -> pd.DataFrame:
    query = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": ponto["lon"],
        "latitude": ponto["lat"],
        "start": START,
        "end": END,
        "format": "JSON",
        "time-standard": "UTC",
    }

    print(f"Baixando NASA POWER - {ponto['bioma']} ({ponto['cidade_ref']})")
    resp = requests.get(BASE_URL, params=query, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    parametros = payload.get("properties", {}).get("parameter", {})
    if not parametros:
        raise ValueError(
            "A resposta da NASA POWER não veio no formato esperado. "
            f"Resposta inicial: {str(payload)[:500]}"
        )

    # Cada parâmetro vem como um dicionário: {"20250101": valor, "20250102": valor, ...}
    dfs = []
    for parametro, serie in parametros.items():
        temp = pd.DataFrame({
            "data": pd.to_datetime(list(serie.keys()), format="%Y%m%d"),
            parametro: list(serie.values()),
        })
        dfs.append(temp)

    df = dfs[0]
    for temp in dfs[1:]:
        df = df.merge(temp, on="data", how="outer")

    df = df.rename(columns=RENOMEAR)

    df["bioma"] = ponto["bioma"]
    df["cidade_ref"] = ponto["cidade_ref"]
    df["lat_ref"] = ponto["lat"]
    df["lon_ref"] = ponto["lon"]

    # Garante que as colunas numéricas sejam números.
    colunas_numericas = [
        "temperatura_c", "temperatura_max_c", "temperatura_min_c",
        "umidade_pct", "precipitacao_mm", "vento_ms"
    ]
    for col in colunas_numericas:
        if col not in df.columns:
            raise KeyError(f"A NASA POWER não retornou a coluna esperada: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Converte vento de m/s para km/h para facilitar apresentação.
    df["vento_kmh"] = df["vento_ms"] * 3.6

    return df[[
        "data", "bioma", "cidade_ref", "lat_ref", "lon_ref",
        "temperatura_c", "temperatura_max_c", "temperatura_min_c",
        "umidade_pct", "precipitacao_mm", "vento_ms", "vento_kmh"
    ]].sort_values("data")


def main():
    partes = [baixar_ponto(p) for p in PONTOS_BIOMAS]
    clima = pd.concat(partes, ignore_index=True)

    saida = OUT_DIR / "nasa_power_biomas_2025.csv"
    clima.to_csv(saida, index=False, encoding="utf-8")

    print(f"\nArquivo salvo em: {saida}")
    print(clima.head())
    print(f"Linhas: {len(clima)} | Biomas: {clima['bioma'].nunique()}")


if __name__ == "__main__":
    main()
