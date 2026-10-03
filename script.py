"""
ACH2118 - Introdução ao Processamento de Língua Natural
EP1 - Classificação de clareza de respostas do e-SIC ({c1, c234, c5})

Pipeline de experimentação: baseline rápido, várias famílias de algoritmos,
grid search com validação cruzada e checagem de overfitting antes de gerar
a planilha final de teste rotulado.

Uso:
    python ep1_pipeline.py --quick                 # só o baseline, resultado rápido
    python ep1_pipeline.py --full                  # grid search completo em todos os modelos
    python ep1_pipeline.py --full --models lr svm  # grid search só nos modelos escolhidos
    python ep1_pipeline.py --predict-only          # usa o melhor modelo já salvo (best_model.joblib)

Arquivos esperados no mesmo diretório (ou informe o caminho via --train/--test):
    train.xlsx  -> colunas: resp_text, clarity  (clarity em {c1, c234, c5})
    test1.xlsx  -> colunas: resp_text, clarity  (clarity vazio, a ser preenchido)

Saída:
    test1_predicted.xlsx  -> mesma planilha de teste, com a coluna 'clarity' preenchida
    cv_results.csv         -> tabela comparativa dos modelos (para o relatório)
    best_model.joblib      -> pipeline treinado vencedor (features + classificador)
"""

import argparse
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LABELS = ["c1", "c234", "c5"]

# Teto de vocabulário do TF-IDF. Sem isso, em 20k documentos o vocabulário
# facilmente passa de 100k termos, e a regressão logística/MLP/etc. demoram
# MUITO mais para convergir. Ajustável via --max-features.
MAX_FEATURES = 50_000


# --------------------------------------------------------------------------
# 1. Carga e limpeza de texto
# --------------------------------------------------------------------------

def load_data(train_path: str, test_path: str):
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    assert list(train.columns[:2]) == ["resp_text", "clarity"], (
        f"Colunas inesperadas em {train_path}: {list(train.columns)}"
    )
    assert list(test.columns[:2]) == ["resp_text", "clarity"], (
        f"Colunas inesperadas em {test_path}: {list(test.columns)}"
    )
    train = train.dropna(subset=["resp_text", "clarity"]).reset_index(drop=True)
    return train, test


_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CTRL_RE = re.compile(r"[\x80-\x9f]")  # bytes de encoding quebrado (ex.: \x96, \x93)


def clean_text(text: str) -> str:
    """Normalização leve: preserva pontuação/acentos, remove ruído de encoding."""
    if not isinstance(text, str):
        return ""
    text = _CTRL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# 2. Features numéricas extras (além de TF-IDF) - úteis para "clareza"
#    Hipótese: respostas confusas tendem a ser mais longas, com mais links,
#    mais siglas/maiúsculas e menos pontuação organizadora.
# --------------------------------------------------------------------------

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """Extrai estatísticas superficiais do texto como features numéricas."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = text if isinstance(text, str) else ""
            n_chars = len(text)
            words = text.split()
            n_words = len(words) or 1
            n_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
            n_urls = len(_URL_RE.findall(text))
            n_digits = sum(ch.isdigit() for ch in text)
            n_upper = sum(ch.isupper() for ch in text)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
            rows.append([
                n_chars,
                n_words,
                n_chars / n_words,          # densidade caractere/palavra
                n_words / n_sentences,      # palavras por "sentença"
                n_urls,
                n_digits / (n_chars or 1),
                n_upper / (n_chars or 1),
                avg_word_len,
            ])
        return np.array(rows, dtype=float)


def build_text_stats_pipeline():
    return Pipeline([
        ("stats", TextStatsExtractor()),
        ("scale", StandardScaler()),
    ])


# --------------------------------------------------------------------------
# 3. Definição dos modelos candidatos
#    Cada entrada: nome -> (pipeline sklearn, grade de hiperparâmetros)
# --------------------------------------------------------------------------

def word_tfidf():
    return TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


def char_tfidf():
    # n-gramas de caractere ajudam com erros de digitação, siglas e morfologia do PT-BR
    return TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.9,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


def combined_features():
    """FeatureUnion: TF-IDF de palavras + TF-IDF de caracteres + estatísticas do texto."""
    return FeatureUnion([
        ("word_tfidf", word_tfidf()),
        ("char_tfidf", char_tfidf()),
        ("stats", Pipeline([
            ("select", FunctionTransformer(lambda X: np.asarray(X))),
            ("extract", build_text_stats_pipeline()),
        ])),
    ])


def get_model_configs(quick: bool = False):
    """Retorna dict {nome: (pipeline, param_grid)}.

    Se quick=True, as grades são reduzidas a 1 combinação (para teste rápido).
    """
    configs = {}

    # --- BASELINE OFICIAL: TF-IDF + Regressão Logística (obrigatório para comparação) ---
    configs["baseline_lr"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE))]),
        {} if quick else {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 3, 5],
            "clf__C": [0.1, 1, 3, 10],
            "clf__class_weight": [None, "balanced"],
        },
    )

    # --- Naive Bayes (rápido, bom baseline de texto) ---
    configs["naive_bayes"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", ComplementNB())]),
        {} if quick else {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__alpha": [0.1, 0.5, 1.0, 2.0],
        },
    )

    # --- SVM linear (costuma ser forte em texto de alta dimensionalidade) ---
    configs["linear_svm"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(
            random_state=RANDOM_STATE, dual="auto"))]),
        {} if quick else {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__C": [0.1, 0.5, 1, 3, 10],
            "clf__class_weight": [None, "balanced"],
        },
    )

    # --- SGD (log_loss / hinge) - treina rápido mesmo em bases grandes ---
    configs["sgd"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", SGDClassifier(
            random_state=RANDOM_STATE, max_iter=2000, tol=1e-3))]),
        {} if quick else {
            "clf__loss": ["hinge", "log_loss", "modified_huber"],
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__penalty": ["l2", "elasticnet"],
        },
    )

    # --- Random Forest sobre TF-IDF ---
    configs["random_forest"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1))]),
        {} if quick else {
            "tfidf__max_df": [0.9],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 40],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    )

    # --- Gradient boosting (HistGradientBoosting exige entrada densa -> reduz
    #     a dimensionalidade do TF-IDF esparso com SVD antes de treinar) ---
    configs["hist_gradient_boosting"] = (
        Pipeline([
            ("tfidf", word_tfidf()),
            ("svd", TruncatedSVD(n_components=150, random_state=RANDOM_STATE)),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        {} if quick else {
            "svd__n_components": [100, 200],
            "clf__max_iter": [150, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
        },
    )

    # --- MLP (rede neural rasa) sobre TF-IDF - alternativa não linear
    #     (early_stopping=False: em algumas versões do sklearn a validação
    #     interna de early stopping quebra com rótulos em string) ---
    configs["mlp"] = (
        Pipeline([("tfidf", word_tfidf()), ("clf", MLPClassifier(
            random_state=RANDOM_STATE, max_iter=300, early_stopping=False))]),
        {} if quick else {
            "clf__hidden_layer_sizes": [(100,), (128, 32)],
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3, 5e-4],
        },
    )

    # --- Features combinadas (palavra + char n-gram + estatísticas) + SVM linear ---
    configs["combined_features_svm"] = (
        Pipeline([("features", combined_features()), ("clf", LinearSVC(
            random_state=RANDOM_STATE, dual="auto"))]),
        {} if quick else {
            "clf__C": [0.3, 1, 3],
        },
    )

    # --- Ensemble por stacking (combina LR, SVM e NB com um meta-classificador) ---
    if not quick:
        base_estimators = [
            ("lr", Pipeline([("tfidf", word_tfidf()), ("clf", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE))])),
            ("svm", Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(
                random_state=RANDOM_STATE, dual="auto"))])),
            ("nb", Pipeline([("tfidf", word_tfidf()), ("clf", ComplementNB())])),
        ]
        configs["stacking_ensemble"] = (
            StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=2000),
                n_jobs=-1,
            ),
            {},  # os hiperparâmetros já foram ajustados nos modelos-base individualmente
        )

    return configs


# --------------------------------------------------------------------------
# 3b. Modelo opcional "inovador": embeddings semânticos (sentence-transformers)
#     TF-IDF/BoW + LR/SVM/NB não pontuam no critério de inovação do enunciado;
#     esta representação é semântica/densa, não bag-of-words. É opcional porque
#     depende de baixar um modelo pré-treinado (precisa de internet na 1a vez).
# --------------------------------------------------------------------------

def try_build_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[info] 'sentence-transformers' não instalado - pulando modelo de "
              "embeddings semânticos. Para habilitar: pip install sentence-transformers")
        return None

    class SentenceEmbedder(BaseEstimator, TransformerMixin):
        def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
            self.model_name = model_name

        def fit(self, X, y=None):
            self._model = SentenceTransformer(self.model_name)
            return self

        def transform(self, X):
            texts = [clean_text(t) for t in X]
            return self._model.encode(texts, show_progress_bar=False, batch_size=64)

    pipeline = Pipeline([
        ("embed", SentenceEmbedder()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    grid = {"clf__C": [0.5, 1, 3]}
    return pipeline, grid


# --------------------------------------------------------------------------
# 4. Avaliação: baseline de classe majoritária, grid search + CV, overfitting
# --------------------------------------------------------------------------

def majority_baseline_accuracy(y_train, y_holdout):
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    preds = dummy.predict(np.zeros((len(y_holdout), 1)))
    return accuracy_score(y_holdout, preds)


def run_quick_baseline(X_train, y_train, X_hold, y_hold):
    """'Comece simples': roda só o baseline oficial rapidamente antes de tudo."""
    print("\n=== BASELINE RÁPIDO (comece simples) ===")
    t0 = time.time()
    maj_acc = majority_baseline_accuracy(y_train, y_hold)
    print(f"Baseline de classe majoritária: acc={maj_acc:.4f}")

    pipe = Pipeline([("tfidf", word_tfidf()), ("clf", LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    hold_acc = accuracy_score(y_hold, pipe.predict(X_hold))
    print(f"Baseline oficial TF-IDF+LR: treino_acc={train_acc:.4f}  "
          f"holdout_acc={hold_acc:.4f}  (tempo={time.time() - t0:.1f}s)")
    print("(o resultado final deve superar tanto o baseline majoritário quanto este)")
    return maj_acc, hold_acc


def run_grid_search(name, pipeline, grid, X, y, cv, search_mode="random",
                     n_iter=12, verbose=1):
    """search_mode='grid' testa TODAS as combinações (pode ser muito lento em
    bases grandes); 'random' sorteia até n_iter combinações, o que limita o
    tempo total de forma previsível sem abrir mão de testar vários parâmetros.
    """
    t0 = time.time()
    if grid:
        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        if search_mode == "grid" or n_combinations <= n_iter:
            search = GridSearchCV(
                pipeline, grid, cv=cv, scoring="accuracy", n_jobs=-1,
                verbose=verbose, refit=True,
            )
        else:
            search = RandomizedSearchCV(
                pipeline, grid, n_iter=n_iter, cv=cv, scoring="accuracy",
                n_jobs=-1, verbose=verbose, refit=True,
                random_state=RANDOM_STATE,
            )
        search.fit(X, y)
        best_estimator = search.best_estimator_
        mean_cv = search.best_score_
        std_cv = search.cv_results_["std_test_score"][search.best_index_]
        best_params = search.best_params_
    else:
        # sem grade (ex.: stacking já ajustado) -> apenas cross_val_score manual
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        pipeline.fit(X, y)
        best_estimator = pipeline
        mean_cv = scores.mean()
        std_cv = scores.std()
        best_params = {}
    elapsed = time.time() - t0
    print(f"[{name}] CV acc={mean_cv:.4f} (+/-{std_cv:.4f})  tempo={elapsed:.1f}s  "
          f"params={best_params}")
    return {
        "model": name,
        "cv_accuracy_mean": mean_cv,
        "cv_accuracy_std": std_cv,
        "fit_time_s": elapsed,
        "best_params": str(best_params),
        "estimator": best_estimator,
    }


def check_overfitting(estimator, X_train, y_train, X_hold, y_hold, name):
    train_acc = accuracy_score(y_train, estimator.predict(X_train))
    hold_acc = accuracy_score(y_hold, estimator.predict(X_hold))
    gap = train_acc - hold_acc
    flag = "  <-- possível overfitting (gap > 0.08)" if gap > 0.08 else ""
    print(f"[{name}] treino_acc={train_acc:.4f}  holdout_acc={hold_acc:.4f}  "
          f"gap={gap:.4f}{flag}")
    return train_acc, hold_acc, gap


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    global MAX_FEATURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.xlsx")
    parser.add_argument("--test", default="test1.xlsx")
    parser.add_argument("--output", default="test1_predicted.xlsx")
    parser.add_argument("--quick", action="store_true",
                         help="só roda o baseline rápido (sem grid search)")
    parser.add_argument("--full", action="store_true",
                         help="roda grid search completo em todos os modelos")
    parser.add_argument("--models", nargs="*", default=None,
                         help="lista de modelos a rodar (default: todos)")
    parser.add_argument("--with-embeddings", action="store_true",
                         help="inclui o modelo opcional de embeddings semânticos "
                              "(precisa de sentence-transformers + internet)")
    parser.add_argument("--n-splits", type=int, default=5,
                         help="número de folds na validação cruzada")
    parser.add_argument("--search", choices=["grid", "random"], default="random",
                         help="'random' (default, rápido/previsível) testa até "
                              "--n-iter combinações sorteadas; 'grid' testa TODAS "
                              "as combinações (pode demorar muito em bases grandes)")
    parser.add_argument("--n-iter", type=int, default=12,
                         help="nº de combinações testadas por modelo quando "
                              "--search=random (default: 12)")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES,
                         help="teto de vocabulário do TF-IDF (reduz para acelerar)")
    parser.add_argument("--predict-only", action="store_true",
                         help="pula o treino e usa best_model.joblib salvo")
    args = parser.parse_args()
    MAX_FEATURES = args.max_features

    train_df, test_df = load_data(args.train, args.test)
    X_all = train_df["resp_text"].astype(str).values
    y_all = train_df["clarity"].astype(str).values

    print(f"Treino: {len(X_all)} exemplos | distribuição:")
    print(pd.Series(y_all).value_counts())

    # split de holdout (fora da CV) só para a checagem final de overfitting
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE
    )

    if args.predict_only:
        best_estimator = joblib.load("best_model.joblib")
        best_name = "modelo_salvo"
    else:
        run_quick_baseline(X_tr, y_tr, X_hold, y_hold)

        if args.quick:
            print("\n--quick: encerrando aqui. Rode com --full para o grid search completo.")
            return

        cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
        configs = get_model_configs(quick=not args.full)

        if args.with_embeddings:
            emb = try_build_embedding_model()
            if emb is not None:
                configs["semantic_embeddings_lr"] = emb

        if args.models:
            configs = {k: v for k, v in configs.items() if k in args.models}

        print(f"\n=== GRID SEARCH + VALIDAÇÃO CRUZADA ({len(configs)} modelos, "
              f"{args.n_splits} folds) ===")
        results = []
        for name, (pipeline, grid) in configs.items():
            try:
                res = run_grid_search(name, pipeline, grid, X_tr, y_tr, cv,
                                       search_mode=args.search, n_iter=args.n_iter)
                results.append(res)
            except Exception as e:
                print(f"[{name}] FALHOU: {e}")

        results_df = pd.DataFrame(results).sort_values(
            "cv_accuracy_mean", ascending=False)
        results_df.drop(columns=["estimator"]).to_csv("cv_results2.csv", index=False)
        print("\n=== RANKING (validação cruzada no conjunto de treino) ===")
        print(results_df.drop(columns=["estimator"]).to_string(index=False))

        best_row = results_df.iloc[0]
        best_name = best_row["model"]
        best_estimator = best_row["estimator"]

        print(f"\nMelhor modelo por CV: {best_name} "
              f"(acc={best_row['cv_accuracy_mean']:.4f})")

        print("\n=== CHECAGEM DE OVERFITTING (holdout separado da CV) ===")
        check_overfitting(best_estimator, X_tr, y_tr, X_hold, y_hold, best_name)

        # refit no conjunto de treino completo (treino + holdout) antes de gerar o teste
        print("\nRe-treinando o melhor modelo com 100% dos dados de treino...")
        best_estimator.fit(X_all, y_all)
        joblib.dump(best_estimator, "best_model.joblib")

    # --------------------------------------------------------------------
    # 6. Predição no conjunto de teste e geração da planilha final
    # --------------------------------------------------------------------
    X_test = test_df["resp_text"].astype(str).values
    preds = best_estimator.predict(X_test)

    out_df = test_df.copy()
    out_df["clarity"] = preds  # mesma estrutura de linhas/colunas do arquivo original
    out_df.to_excel(args.output, index=False)
    print(f"\nPlanilha de teste rotulada salva em: {args.output} "
          f"({len(out_df)} linhas, formato idêntico ao original)")
    print("Distribuição prevista no teste:")
    print(pd.Series(preds).value_counts())


if __name__ == "__main__":
    main()