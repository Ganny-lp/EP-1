# EP1 — ACH2118 — Classificação de clareza de respostas do e-SIC

**Arquivo final a ser analisado: `script.py`**

Este README documenta exatamente o que `script.py` faz, como rodar, o que ele
produz e como interpretar a saída — para uso no relatório e na correção.

---

## 1. O problema

Classificação ternária de texto: dado o texto de uma resposta de e-SIC
(`resp_text`), prever o rótulo de clareza `clarity ∈ {c1, c234, c5}`.

- `train.xlsx` — 20.092 exemplos rotulados (colunas `resp_text`, `clarity`).
- `test1.xlsx` — exemplos sem rótulo (mesmas colunas, `clarity` vazio).
- Distribuição das classes no treino: `c5`=6.892, `c234`=6.853, `c1`=6.347
  (razoavelmente balanceada).

## 2. Requisitos

```bash
pip install pandas numpy scipy scikit-learn joblib openpyxl
```

`sentence-transformers` é **opcional**, só necessário se quiser usar
`--with-embeddings` (ver seção 7). Sem ele instalado, o script simplesmente
avisa e pula esse modelo — não quebra a execução.

## 3. Arquivos de entrada esperados

Por padrão, `script.py` procura `train.xlsx` e `test1.xlsx` no diretório
atual. Caminhos diferentes podem ser passados com `--train` e `--test`.
As duas primeiras colunas de cada planilha precisam se chamar exatamente
`resp_text` e `clarity` (o script valida isso e falha cedo se não bater).

## 4. Como rodar

### 4.1. Passo 1 — baseline rápido (sempre rode isso primeiro)

```bash
python script.py --quick
```

Treina **só** o baseline oficial (TF-IDF + Regressão Logística, sem busca de
hiperparâmetros) e imprime:
- a acurácia do baseline de classe majoritária (referência mínima absoluta);
- a acurácia de treino vs. holdout do baseline oficial.

Isso serve para (a) confirmar que os dados carregam certo e (b) ter uma
primeira noção de overfitting antes de gastar tempo com busca de
hiperparâmetros. Não grava nenhum arquivo de resultado — é só um
diagnóstico no terminal.

### 4.2. Passo 2 — comparação completa entre algoritmos

```bash
python script.py --full
```

Roda busca de hiperparâmetros com validação cruzada (`StratifiedKFold`,
5 folds por padrão) em **9 famílias de modelos** (lista completa na seção 5),
compara todos por acurácia média de CV, escolhe o melhor, reavalia overfitting
num holdout separado, re-treina com 100% do treino e gera a planilha de teste
rotulada.

Para rodar só alguns modelos (mais rápido, útil para testar):

```bash
python script.py --full --models baseline_lr naive_bayes linear_svm sgd
```

Nomes válidos para `--models`: `baseline_lr`, `naive_bayes`, `linear_svm`,
`sgd`, `random_forest`, `hist_gradient_boosting`, `mlp`,
`combined_features_svm`, `stacking_ensemble` (e `semantic_embeddings_lr` se
`--with-embeddings` estiver ativo).

### 4.3. Flags importantes de desempenho

| Flag | Default | Para que serve |
|---|---|---|
| `--search {grid,random}` | `random` | `random` sorteia até `--n-iter` combinações de hiperparâmetros por modelo (tempo previsível); `grid` testa **todas** as combinações (pode demorar horas em bases grandes — só use se tiver tempo de sobra ou máquina rápida). |
| `--n-iter N` | `12` | Nº de combinações testadas por modelo quando `--search=random`. |
| `--n-splits N` | `5` | Nº de folds na validação cruzada. Reduzir para `3` acelera bastante com pouca perda de confiabilidade. |
| `--max-features N` | `50000` | Teto do vocabulário do TF-IDF. Reduzir (ex. `20000`) acelera a convergência dos modelos lineares/redes, às custas de um pouco de informação. |
| `--results-csv ARQUIVO` | `cv_results.csv` | Onde salvar a tabela comparativa dos modelos. |

Se a execução estiver demorando demais na sua máquina, o caminho mais rápido
para ter *algum* resultado comparativo decente é:

```bash
python script.py --full --models baseline_lr naive_bayes linear_svm sgd \
    --n-splits 3 --n-iter 10
```

### 4.4. Reaproveitar o modelo já treinado

```bash
python script.py --predict-only --test test1.xlsx
```

Carrega `best_model.joblib` (salvo na execução anterior) e só gera a
planilha de predição, sem re-treinar nada.

## 5. Modelos avaliados

| Nome interno | Algoritmo | Observação |
|---|---|---|
| `baseline_lr` | TF-IDF (palavras) + Regressão Logística | **Baseline oficial** exigido no enunciado — referência obrigatória de comparação. |
| `naive_bayes` | TF-IDF + Complement Naive Bayes | Rápido, bom ponto de partida para texto. |
| `linear_svm` | TF-IDF + SVM linear (`LinearSVC`) | Costuma performar bem em espaços de alta dimensionalidade como TF-IDF. |
| `sgd` | TF-IDF + `SGDClassifier` (hinge/log/modified_huber) | Treina rápido mesmo com a base inteira. |
| `random_forest` | TF-IDF + Random Forest | Não linear, captura interações entre termos. |
| `hist_gradient_boosting` | TF-IDF → SVD (redução de dimensionalidade) + Gradient Boosting | O boosting do sklearn exige entrada densa, por isso há uma etapa de SVD antes dele. |
| `mlp` | TF-IDF + rede neural rasa (`MLPClassifier`) | Alternativa não linear baseada em rede neural. |
| `combined_features_svm` | (TF-IDF de palavras + TF-IDF de caracteres 3–5 + estatísticas do texto) + SVM linear | Combina representações diferentes: n-gramas de caractere ajudam com erros de digitação/morfologia do PT-BR; as estatísticas (tamanho do texto, nº de URLs, proporção de maiúsculas/dígitos etc.) capturam sinais de formatação que o TF-IDF puro não vê. |
| `stacking_ensemble` | Ensemble por stacking de LR + SVM + NB, com meta-classificador logístico | Combina as previsões dos três modelos-base por um classificador supervisionado em cima. |
| `semantic_embeddings_lr` *(opcional, `--with-embeddings`)* | Embeddings semânticos multilíngues (`sentence-transformers`) + Regressão Logística | Representação densa/semântica, não bag-of-words — diferente das demais em natureza, não só no classificador. |

As **estatísticas textuais** usadas em `combined_features_svm` partem da
hipótese de que respostas pouco claras tendem a ser mais longas, com mais
links e menos pontuação organizadora — por isso entram: nº de caracteres,
nº de palavras, caracteres por palavra, palavras por "sentença" (separada
por `.`/`!`/`?`), nº de URLs, proporção de dígitos, proporção de
maiúsculas e tamanho médio de palavra.

## 6. Metodologia de avaliação (o que olhar no relatório)

1. **Baseline de classe majoritária**: referência mínima absoluta — qualquer
   modelo abaixo disso está pior que simplesmente "chutar a classe mais
   comum". Calculado via `DummyClassifier(strategy="most_frequent")`.
2. **Baseline oficial (TF-IDF + Regressão Logística)**: referência de
   comparação exigida no enunciado; o modelo final deve superá-la.
3. **Validação cruzada estratificada** (`StratifiedKFold`) em todos os
   modelos candidatos, preservando a proporção das 3 classes em cada fold.
4. **Busca de hiperparâmetros** (grid ou randomizada) por modelo, conforme
   a grade definida em `get_model_configs()`.
5. **Checagem de overfitting**: depois de escolher o melhor modelo por CV,
   ele é reavaliado num *holdout* de 15% separado **antes** da validação
   cruzada (nunca visto durante a busca de hiperparâmetros). O script
   reporta `treino_acc`, `holdout_acc` e o `gap` entre os dois, sinalizando
   quando o gap passa de `0.08` como possível overfitting.
6. **Re-treino final**: o melhor modelo é re-treinado com 100% dos dados de
   treino (treino + holdout) antes de gerar as predições do teste, para não
   desperdiçar dados na versão final.

## 7. Saídas geradas

| Arquivo | Conteúdo |
|---|---|
| `cv_results.csv` | Uma linha por modelo rodado com sucesso: `model`, `cv_accuracy_mean`, `cv_accuracy_std`, `fit_time_s`, `best_params`. Ordenado da maior para a menor acurácia. Gravado **incrementalmente** — se um modelo falhar no meio da execução, os resultados dos modelos anteriores não se perdem. |
| `best_model.joblib` | Pipeline completo (vetorização + classificador) do modelo vencedor, já re-treinado com 100% do treino. Pode ser recarregado com `--predict-only`. |
| `test1_predicted.xlsx` (ou nome passado em `--output`) | Cópia exata de `test1.xlsx` com a coluna `clarity` preenchida — mesma estrutura de linhas/colunas do arquivo original, para não cair na penalidade de "formato inválido do arquivo de teste". |

Se algum modelo falhar durante `--full`, o traceback completo é impresso no
console (prefixado por `[nome_do_modelo] FALHOU`), e a execução continua
para os demais modelos — uma falha isolada não derruba a comparação inteira.

## 8. Limitações conhecidas / decisões assumidas

- `--search random` (padrão) **não** garante encontrar o ótimo global de
  hiperparâmetros — é uma amostragem; para a busca exaustiva, usar
  `--search grid` (mais lento).
- `hist_gradient_boosting` passa por uma redução de dimensionalidade (SVD)
  porque o `HistGradientBoostingClassifier` do scikit-learn não aceita
  matriz esparsa como entrada; isso descarta parte da informação do TF-IDF
  original.
- O modelo de embeddings semânticos (`--with-embeddings`) baixa um modelo
  pré-treinado da internet na primeira execução — não roda em ambiente sem
  acesso à rede.
- `max_features=50000` no TF-IDF é um teto de desempenho/tempo, não um valor
  obtido por otimização — pode ser ajustado via `--max-features` conforme o
  tempo disponível.
