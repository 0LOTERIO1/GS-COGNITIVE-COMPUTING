# Relatório Técnico — FireGuard Orbit

**GS — Cognitive Computing, Computer Vision and IoT Systems | FIAP**  
**Tema:** Space Connect | Prevenção de Queimadas com Machine Learning  
**Tipo de solução:** Ciência de Dados / Machine Learning

---

## 1. Problema

Queimadas representam um dos principais vetores de degradação ambiental no Brasil, afetando biomas como Cerrado, Amazônia, Pantanal, Caatinga, Mata Atlântica e Pampa. A detecção antecipada de áreas com maior risco pode ajudar órgãos ambientais, defesa civil e equipes de monitoramento a priorizar ações preventivas.

**Tipo de problema:** classificação binária supervisionada com dados tabulares ambientais.  
**Variável alvo:** `risco_alto_queimada` — 1 = risco alto; 0 = risco normal.  
**Período analisado na base final:** 2025-01-01 a 2025-05-30.

---

## 2. Metodologia

### 2.1 Base de dados final

A versão final do projeto utiliza uma base integrada com **focos reais de queimadas** e **dados meteorológicos reais**.

| Componente | Fonte / arquivo | Papel no projeto |
|---|---|---|
| Focos de queimadas | INPE / `data/focos_inpe_agregado_2025.csv` | Base principal para o histórico de focos |
| Clima diário | NASA POWER / `data/nasa_power/nasa_power_biomas_2025.csv` | Temperatura, umidade, precipitação e vento reais |
| Base final | `data/dados_fireguard_orbit_real.csv` | Base usada no treinamento e avaliação |

A base final possui **2,099 linhas**, **28 colunas**, **6 biomas**, **5 regiões** e **285,465 focos observados** após a integração e criação das variáveis históricas.

A melhoria em relação à versão inicial é que as variáveis climáticas usadas pelo modelo deixaram de ser demonstrativas e passaram a vir da NASA POWER. Os dados climáticos foram associados por ponto de referência de cada bioma.

### 2.2 Variáveis do modelo

| Tipo | Variáveis |
|---|---|
| Categóricas | `regiao`, `bioma` |
| Numéricas meteorológicas | `temperatura_c`, `temperatura_max_c`, `temperatura_min_c`, `umidade_pct`, `precipitacao_mm`, `vento_kmh` |
| Numéricas históricas | `dias_sem_chuva`, `chuva_acumulada_7d`, `umidade_media_7d`, `temperatura_media_7d`, `focos_dia_anterior`, `focos_acumulados_7d`, `media_focos_7d`, `media_focos_15d` |
| Temporais | `mes`, `dia_do_ano` |

A variável alvo foi definida a partir do comportamento do próximo dia/período da mesma combinação `regiao + bioma`. Para evitar um único limiar nacional, o risco alto foi calculado por bioma, considerando o quartil superior dos focos do próximo dia.

### 2.3 Modelos testados

Foram avaliados 9 algoritmos: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, HistGradientBoosting, KNN e SVM.

### 2.4 Protocolo experimental

- `train_test_split` 80/20 com `stratify=y`.
- Mesmo conjunto de treino e teste para todos os modelos.
- OneHotEncoder para `regiao` e `bioma`.
- StandardScaler para variáveis numéricas.
- Métricas: accuracy, precision, recall, F1-score, ROC-AUC e matriz de confusão.

### 2.5 Critério de seleção

Como o objetivo é prevenção, o critério principal foi o **recall da classe risco alto**. Essa métrica reduz falsos negativos, que representam áreas críticas não detectadas.

Ordem de desempate: F1-score → ROC-AUC.

---

## 3. Resultados

### 3.1 Modelo final selecionado: **SVM**

| Métrica | Valor |
|---|---:|
| Accuracy | 0.7690 |
| Precision — risco alto | 0.5302 |
| **Recall — risco alto** | **0.7453** |
| F1-score — risco alto | 0.6196 |
| ROC-AUC | 0.8430 |

### 3.2 Benchmark comparativo — top 3 por recall

| Modelo | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|
| **SVM** | 0.7453 | 0.6196 | 0.8430 |
| Logistic Regression | 0.6698 | 0.5703 | 0.8261 |
| HistGradientBoosting | 0.6226 | 0.6502 | 0.8578 |

Tabela completa: `benchmark_modelos_real.csv`.

### 3.3 Matriz de confusão

A matriz de confusão do modelo final foi:

| | Predito normal | Predito risco alto |
|---|---:|---:|
| Real normal | 244 | 70 |
| Real risco alto | 27 | 79 |

Interpretação:

- **Verdadeiros positivos:** áreas/períodos de risco alto corretamente sinalizados.
- **Falsos negativos:** áreas/períodos críticos não detectados; erro mais grave para a proposta.
- **Falsos positivos:** alertas preventivos extras; custo mais aceitável no contexto de prevenção.

---

## 4. Análise

Os gráficos exploratórios e de avaliação mostram:

- **Série temporal:** variação dos focos reais de queimadas no período analisado.
- **Taxa de risco por bioma:** diferença de comportamento entre biomas.
- **Dias sem chuva × focos:** relação entre estiagem e focos observados.
- **Correlação:** relação entre variáveis reais de clima, histórico de focos e alvo.
- **Benchmark:** comparação entre 9 algoritmos com o mesmo protocolo experimental.

O resultado ficou mais realista do que a versão anterior. A métrica de recall não é artificialmente alta, mas a análise fica mais defensável, porque os dados ambientais agora vêm de uma fonte real.

---

## 5. Conclusão

O FireGuard Orbit atende ao objetivo da disciplina porque:

- usa dados reais de focos de queimadas e clima;
- aplica tratamento e integração de bases;
- cria variáveis históricas úteis para modelagem;
- realiza análise exploratória;
- treina e compara modelos de classificação;
- avalia resultados com métricas mensuráveis.

O modelo **SVM** foi selecionado por apresentar o maior recall para a classe de risco alto. A escolha foi guiada pelo problema: em prevenção de queimadas, é melhor gerar alguns alertas extras do que deixar passar uma área crítica.

---

## 6. Limitações

- O período analisado ainda é curto.
- Os dados meteorológicos são reais, mas aproximados por ponto de referência do bioma.
- A base não usa ainda grade climática municipal nem séries históricas longas.
- O modelo é acadêmico e não deve ser usado em produção sem validação com mais anos e meses de seca.

---

## 7. Fontes

- INPE Queimadas: https://data.inpe.br/queimadas/dados-abertos/
- NASA POWER: https://power.larc.nasa.gov/
- Scikit-learn: https://scikit-learn.org/

---

## 8. Frase para defesa oral

> "Na versão final, substituímos as variáveis ambientais simuladas por dados meteorológicos reais da NASA POWER. O modelo usa focos reais do INPE, clima real e variáveis históricas para prever risco alto de queimadas. Testamos 9 algoritmos e escolhemos o modelo pelo recall da classe risco alto, porque o erro mais grave é deixar de sinalizar uma área crítica."

---

## 9. Integrantes

| Nome | RM |
|---|---:|
| PEDRO LOTÉRIO DOS SANTOS | 550909 |
| RAFAEL BUENO VILLELA | 550275 |
| LUCAS THOMAZETTE BENVENUTO | 98048 |

---

*Relatório — FireGuard Orbit | GS FIAP 2026/1*
