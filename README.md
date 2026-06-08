# FireGuard Orbit

**Disciplina:** Cognitive Computing, Computer Vision and IoT Systems | FIAP  
**Tema GS:** Space Connect — Prevenção de Queimadas com Machine Learning  
**Tipo de solução:** Ciência de Dados / Machine Learning

**Repositório GitHub:** https://github.com/0LOTERIO1/GS-COGNITIVE-COMPUTING

---

## 1. Problema

Queimadas são um problema ambiental relevante no Brasil. A detecção antecipada de áreas com maior risco ajuda a priorizar ações preventivas antes que focos se intensifiquem.

## 2. Objetivo

Prever **risco alto de queimada** (`risco_alto_queimada = 1`) por combinação de **data, região e bioma**, usando focos reais de queimadas, dados meteorológicos reais e variáveis históricas derivadas.

## 3. Dados utilizados — versão final real

A versão final do projeto substitui as variáveis ambientais simuladas por dados meteorológicos reais da NASA POWER.

| Componente | Arquivo | Origem / uso |
|---|---|---|
| Focos de queimadas agregados | `data/focos_inpe_agregado_2025.csv` | Focos reais do INPE, agregados por data, região e bioma |
| Dados meteorológicos | `data/nasa_power/nasa_power_biomas_2025.csv` | NASA POWER, com pontos de referência por bioma |
| Base final do modelo | `data/dados_fireguard_orbit_real.csv` | União dos focos com clima real e variáveis históricas |

**Resumo da base final:**

- Período modelado: **2025-01-01 a 2025-05-30**
- Linhas: **2,099**
- Colunas: **28**
- Biomas: **6**
- Regiões: **5**
- Focos observados na base final: **285,465**
- Taxa da classe risco alto: **25.2%**

> Observação metodológica: os dados meteorológicos são reais, mas associados a **pontos de referência por bioma**. Isso torna a base mais séria que a versão simulada, mas ainda não substitui uma grade climática municipal completa.

## 4. Fontes

- INPE Queimadas: https://data.inpe.br/queimadas/dados-abertos/
- NASA POWER: https://power.larc.nasa.gov/
- Scikit-learn: https://scikit-learn.org/

A NASA FIRMS foi mantida como referência de expansão futura, mas a versão final entregue usa **INPE + NASA POWER**.

## 5. Metodologia

1. Consolidação dos focos reais de queimadas por data, região e bioma.
2. Coleta de dados meteorológicos reais da NASA POWER para pontos de referência dos biomas.
3. Integração das bases por `data` e `bioma`.
4. Criação de variáveis históricas reais, como:
   - `focos_dia_anterior`
   - `focos_acumulados_7d`
   - `media_focos_7d`
   - `media_focos_15d`
   - `chuva_acumulada_7d`
   - `dias_sem_chuva`
   - `umidade_media_7d`
   - `temperatura_media_7d`
5. Definição da variável alvo: risco alto no próximo dia/período da mesma região e bioma.
6. Benchmark com 9 algoritmos, usando a mesma divisão treino/teste.
7. Seleção do modelo com maior **recall da classe risco alto**, com F1-score e ROC-AUC como desempate.

## 6. Modelos avaliados

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- HistGradientBoosting
- KNN
- SVM

## 7. Resultados — modelo final

O modelo selecionado foi **SVM**, porque teve o maior recall para a classe de risco alto.

| Métrica | Valor |
|---|---:|
| Accuracy | 0.7690 |
| Precision — risco alto | 0.5302 |
| **Recall — risco alto** | **0.7453** |
| F1-score — risco alto | 0.6196 |
| ROC-AUC | 0.8430 |

**Critério de seleção:** priorizar recall da classe risco alto. Em prevenção de queimadas, um falso negativo representa uma área crítica não detectada, o que é mais grave do que gerar um alerta preventivo extra.

Arquivos de resultado:

- `benchmark_modelos_real.csv`
- `resultados_modelo_real.json`

## 8. Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Para reproduzir a base real e o benchmark:

```bash
python scripts/01_baixar_nasa_power_biomas.py
python scripts/02_montar_base_real_fireguard.py
python scripts/03_treinar_benchmark_base_real.py
```

Para abrir o notebook:

```bash
jupyter notebook GS_FireGuard_Orbit.ipynb
```

## 9. Estrutura principal

```text
GS_Cognitive_FireGuard_Orbit/
  GS_FireGuard_Orbit.ipynb
  README.md
  relatorio_fireguard_orbit.md
  fontes_dados.txt
  benchmark_modelos_real.csv
  resultados_modelo_real.json
  requirements.txt
  data/
    focos_inpe_agregado_2025.csv
    dados_fireguard_orbit_real.csv
    nasa_power/
      nasa_power_biomas_2025.csv
  scripts/
    01_baixar_nasa_power_biomas.py
    02_montar_base_real_fireguard.py
    03_treinar_benchmark_base_real.py
  graficos/
  src/
```

## 10. Limitações

- O período ainda é curto e cobre principalmente janeiro a maio de 2025.
- Os dados meteorológicos são reais, porém usam pontos de referência por bioma, não grade climática para cada município.
- O modelo é acadêmico e não deve ser usado em produção sem validação com séries históricas maiores e meses de seca.

## 11. Integrantes

| Nome | RM |
|---|---:|
| PEDRO LOTÉRIO DOS SANTOS | 550909 |
| RAFAEL BUENO VILLELA | 550275 |
| LUCAS THOMAZETTE BENVENUTO | 98048 |
