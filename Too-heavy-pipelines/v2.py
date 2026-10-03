"""
ACH2118 - Introdução ao Processamento de Língua Natural
EP1 v2 - Classificação de clareza de respostas do e-SIC ({c1, c234, c5})

Pipeline redesenhado a partir do diagnóstico de conflito de rótulos:

  - 499 grupos de textos repetidos / 2.449 linhas nesses grupos;
  - 316 grupos com rótulos conflitantes / 1.977 linhas em conflito;
  - teto de acurácia (majority-label ceiling) desses textos repetidos: 62,39%;
  - features estruturais (tamanho, TTR, URLs, números) diferem sistematicamente
    entre c1 / c234 / c5, sugerindo sinal de classe na FORMA do texto, não só
    no conteúdo lexical.

Consequências de design deste script (em relação ao script.py original):

  1. Texto NORMALIZADO (template canônico): saudações, números, datas, e-mails
     e URLs são substituídos por marcadores ([SAUDACAO], [NUM], [DATA], [EMAIL],
     [URL]) para revelar templates institucionais repetidos.
  2. Diagnóstico de duplicatas/conflitos é recalculado e salvo em CSV, incluindo
     o "grupo canônico" de cada linha (usado como `groups` no GroupKFold).
  3. Representação combinada (4 fontes, com pesos ajustáveis via grid search):
         TFIDF_word(texto original)
       + TFIDF_char(texto original)
       + TFIDF_word(texto normalizado)
       + features estruturais (10 métricas, padronizadas)
       + [opcional] embeddings MiniLM multilíngue (cacheados em disco)
  4. Validação DUPLA em todo modelo:
       - StratifiedKFold (validação "normal")
       - GroupKFold usando o texto normalizado como grupo (nenhum
         texto/template igual fica dividido entre treino e validação)
     A diferença entre as duas revela quanto da acurácia "normal" vem de
     repetição/memorização de templates.
  5. Ensemble lexical + semântico (soft voting) quando o MiniLM está disponível.
  6. Seleção do melhor modelo é feita pela métrica GroupKFold (mais realista),
     não pela StratifiedKFold.

Uso:
    python ep1_pipeline_v2.py --quick
    python ep1_pipeline_v2.py --full
    python ep1_pipeline_v2.py --full --models lexical_svm hybrid_lexical_semantic
    python ep1_pipeline_v2.py --full --with-minilm
    python ep1_pipeline_v2.py --predict-only

Dependência opcional para o bloco semântico (MiniLM):
    pip install sentence-transformers torch

Arquivos esperados:
    train.xlsx -> colunas: resp_text, clarity
    test1.xlsx -> colunas: resp_text, clarity (vazia)

Saídas:
    test1_predicted_v2.xlsx
    template_diagnostico.csv        (grupos repetidos/conflitantes)
    cv_results_v2.csv               (StratifiedKFold x GroupKFold por modelo)
    best_model_v2.joblib
    logs/ep1_pipeline_v2.log
"""

import argparse
import hashlib
import json
import logging
import pickle
import re
import time
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LABELS = ["c1", "c234", "c5"]
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "ep1_pipeline_v2.log"
CACHE_DIR = Path("transformer_cache")
MINILM_CACHE_FILE = CACHE_DIR / "minilm_embeddings_cache.pkl"
logger = logging.getLogger("ep1_pipeline_v2")


def setup_logging(level=logging.INFO):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(fh)
    logger.info("=" * 80)
    logger.info("EP1 v2 - pipeline híbrido (conteúdo + estrutura + semântica)")
    logger.info("Random state: %s", RANDOM_STATE)


# --------------------------------------------------------------------------
# 1. Limpeza e normalização de texto
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x80-\x9f]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+")
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_LONGNUM_RE = re.compile(r"\b\d{3,}\b")

# Saudações/fórmulas administrativas mais frequentes (case-insensitive).
_SAUDACAO_PATTERNS = [
    r"prezad[oa]s?\s*\(?a?\)?\s*(senhor|senhora|cidad[ãa]o?|cidad[ãa]|usuári[oa]|requerente)?",
    r"car[oa]s?\s*\(?a?\)?\s*(senhor|senhora|cidad[ãa]o?|cidad[ãa]|usuári[oa])?",
    r"em\s+resposta\s+(a|à|ao)\s+sua\s+solicita[çc][ãa]o",
    r"em\s+atendimento\s+(a|à|ao)\s+sua\s+solicita[çc][ãa]o",
    r"encaminhamos\s+em\s+anexo",
    r"agradecemos\s+(o\s+)?contato",
]
_SAUDACAO_RE = re.compile("|".join(_SAUDACAO_PATTERNS), flags=re.IGNORECASE)


def clean_text(text) -> str:
    """Limpeza mínima (mantém conteúdo original) usada nos TF-IDF 'originais'."""
    if not isinstance(text, str):
        return ""
    text = _CTRL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_text(text) -> str:
    """Gera o 'texto canônico/template': remove variação que não carrega
    informação semântica (saudações, números, datas, e-mails, URLs), para
    revelar templates institucionais repetidos e alimentar o TF-IDF
    'normalizado'."""
    text = clean_text(text).lower()
    text = _URL_RE.sub(" [URL] ", text)
    text = _EMAIL_RE.sub(" [EMAIL] ", text)
    text = _DATE_RE.sub(" [DATA] ", text)
    text = _LONGNUM_RE.sub(" [NUM] ", text)
    text = _SAUDACAO_RE.sub(" [SAUDACAO] ", text)
    return _WS_RE.sub(" ", text).strip()


def load_data(train_path: str, test_path: str):
    logger.info("Carregando dados: treino=%s | teste=%s", train_path, test_path)
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    assert list(train.columns[:2]) == ["resp_text", "clarity"], train.columns.tolist()
    assert list(test.columns[:2]) == ["resp_text", "clarity"], test.columns.tolist()
    before = len(train)
    train = train.dropna(subset=["resp_text", "clarity"]).reset_index(drop=True)
    logger.info(
        "Treino: %d -> %d linhas após limpeza | Teste: %d linhas", before, len(train), len(test)
    )
    return train, test


# --------------------------------------------------------------------------
# 2. Diagnóstico de duplicatas/conflitos (reproduz e salva a análise)
# --------------------------------------------------------------------------

def diagnose_duplicates(train_df: pd.DataFrame, out_csv="template_diagnostico.csv"):
    norm = train_df["resp_text"].astype(str).map(normalize_text)
    y = train_df["clarity"].astype(str)
    df = pd.DataFrame({"norm_text": norm, "clarity": y})

    group_sizes = df.groupby("norm_text").size()
    duplicate_rows = int((group_sizes[group_sizes > 1]).sum())
    n_repeated_groups = int((group_sizes > 1).sum())

    label_nunique = df.groupby("norm_text")["clarity"].nunique()
    conflict_groups = label_nunique[label_nunique > 1].index
    n_conflict_groups = int(len(conflict_groups))
    conflict_mask = df["norm_text"].isin(conflict_groups)
    n_conflict_rows = int(conflict_mask.sum())

    def majority_ratio(g):
        return g.value_counts(normalize=True).iloc[0]

    repeated_groups = group_sizes[group_sizes > 1].index
    if len(repeated_groups) > 0:
        ceiling = (
            df[df["norm_text"].isin(repeated_groups)]
            .groupby("norm_text")["clarity"]
            .apply(majority_ratio)
            .mean()
        )
    else:
        ceiling = np.nan

    logger.info(
        "Diagnóstico de templates: %d grupos repetidos (%d linhas) | "
        "%d grupos conflitantes (%d linhas) | teto de acurácia por maioria=%.4f",
        n_repeated_groups, duplicate_rows, n_conflict_groups, n_conflict_rows, ceiling,
    )

    # combinação de rótulos por grupo conflitante (ex.: c1+c234, c1+c5, ...)
    combo_counts = {}
    for _, g in df[conflict_mask].groupby("norm_text")["clarity"]:
        combo = "+".join(sorted(g.unique()))
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    if combo_counts:
        logger.info("Combinações de rótulos em grupos conflitantes: %s", combo_counts)

    report = (
        df.groupby("norm_text")
        .agg(n_linhas=("clarity", "size"), n_rotulos=("clarity", "nunique"))
        .sort_values("n_linhas", ascending=False)
    )
    report.to_csv(out_csv, encoding="utf-8-sig")
    logger.info("Diagnóstico completo salvo em: %s", out_csv)

    # grupo canônico por linha, para uso como `groups` no GroupKFold
    group_ids = norm.values
    return group_ids


# --------------------------------------------------------------------------
# 3. Features estruturais (mesmas 10 métricas do diagnóstico)
# --------------------------------------------------------------------------

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for raw in X:
            text = clean_text(raw)
            n_chars = len(text) or 1
            words = text.split()
            n_words = len(words) or 1
            unique_words = len(set(w.lower() for w in words))
            ttr = unique_words / n_words
            n_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
            n_urls = len(_URL_RE.findall(text))
            n_digit_tokens = len(_LONGNUM_RE.findall(text)) + len(re.findall(r"\d", text))
            avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
            n_upper = sum(ch.isupper() for ch in text)
            n_punct = sum(ch in ".,;:!?" for ch in text)
            rows.append([
                n_chars,
                n_words,
                unique_words,
                ttr,
                n_sentences,
                n_words / n_sentences,
                n_urls,
                n_digit_tokens / n_chars,
                n_upper / n_chars,
                (n_punct / n_chars) * 100,
                avg_word_len,
            ])
        return np.asarray(rows, dtype=np.float32)


def _to_object_array(X):
    # função nomeada no nível do módulo (não lambda): precisa ser "picklable"
    # para o cache em disco (joblib.Memory) do Pipeline funcionar.
    return np.asarray(X, dtype=object)


def stats_pipeline():
    return Pipeline([
        ("select", FunctionTransformer(_to_object_array)),
        ("stats", TextStatsExtractor()),
        ("scale", StandardScaler()),
    ])


# --------------------------------------------------------------------------
# 4. Vetorizadores (conteúdo original, morfologia/char, template normalizado)
# --------------------------------------------------------------------------

def word_tfidf_original():
    return TfidfVectorizer(
        preprocessor=clean_text, lowercase=True, ngram_range=(1, 2),
        min_df=3, max_df=0.9, sublinear_tf=True, max_features=40_000,
    )


def char_tfidf_original():
    # char n-gramas (3,5) sem limite de vocabulário explodem para centenas de
    # milhares de colunas em ~20k documentos longos; max_features + min_df
    # maior mantêm os padrões mais informativos e evitam fit/treino muito lentos.
    return TfidfVectorizer(
        preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5),
        min_df=5, max_df=0.9, sublinear_tf=True, max_features=80_000,
    )


def word_tfidf_normalized():
    return TfidfVectorizer(
        preprocessor=normalize_text, lowercase=True, ngram_range=(1, 2),
        min_df=3, max_df=0.9, sublinear_tf=True, max_features=40_000,
    )


def lexical_features():
    """FeatureUnion com as 4 fontes lexicais/estruturais do diagnóstico.
    `transformer_weights` é exposto para grid search (pesos entre fontes)."""
    return FeatureUnion(
        transformer_list=[
            ("word_orig", word_tfidf_original()),
            ("char_orig", char_tfidf_original()),
            ("word_norm", word_tfidf_normalized()),
            ("stats", stats_pipeline()),
        ],
        transformer_weights={"word_orig": 1.0, "char_orig": 1.0, "word_norm": 1.0, "stats": 1.0},
    )


# --------------------------------------------------------------------------
# 5. Embeddings MiniLM (opcional), com cache persistente em disco por texto
# --------------------------------------------------------------------------

class MiniLMEmbedder(BaseEstimator, TransformerMixin):
    """Encoder MiniLM congelado, com cache texto->vetor compartilhado entre
    todas as instâncias/clones (cache de classe) e persistido em disco, para
    nunca reprocessar o mesmo texto duas vezes entre folds/execuções."""

    _mem_cache = {}
    _loaded_from_disk = False

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size

    @classmethod
    def _load_disk_cache(cls):
        if cls._loaded_from_disk:
            return
        cls._loaded_from_disk = True
        if MINILM_CACHE_FILE.exists():
            try:
                with open(MINILM_CACHE_FILE, "rb") as f:
                    cls._mem_cache.update(pickle.load(f))
                logger.info(
                    "[minilm] Cache em disco carregado: %d textos (%s)",
                    len(cls._mem_cache), MINILM_CACHE_FILE,
                )
            except Exception:
                logger.exception("[minilm] Falha ao carregar cache em disco; ignorando")

    @classmethod
    def _save_disk_cache(cls):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MINILM_CACHE_FILE, "wb") as f:
            pickle.dump(cls._mem_cache, f)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._load_disk_cache()
        texts = [clean_text(t) for t in X]
        missing = sorted({t for t in texts if t not in self._mem_cache})
        if missing:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "[minilm] Codificando %d textos novos (de %d) com %s",
                len(missing), len(texts), self.model_name,
            )
            t0 = time.time()
            encoder = SentenceTransformer(self.model_name, device="cpu")
            vecs = encoder.encode(
                missing, batch_size=self.batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
            for t, v in zip(missing, vecs):
                self._mem_cache[t] = v
            self._save_disk_cache()
            logger.info(
                "[minilm] %d textos codificados em %.1fs; cache atualizado (%s)",
                len(missing), time.time() - t0, MINILM_CACHE_FILE,
            )
        return np.vstack([self._mem_cache[t] for t in texts])


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        logger.info(
            "sentence-transformers não instalado; bloco semântico MiniLM será "
            "pulado (pip install sentence-transformers torch para habilitar)."
        )
        return False


# --------------------------------------------------------------------------
# 6. Combinador denso+esparso (para juntar TF-IDF com stats/MiniLM)
# --------------------------------------------------------------------------

class DenseHStack(BaseEstimator, TransformerMixin):
    """Concatena a saída de vários transformers (podendo misturar esparso e
    denso) em uma única matriz esparsa, com pesos por bloco."""

    def __init__(self, blocks, weights=None):
        self.blocks = blocks  # lista de (nome, transformer)
        self.weights = weights or {name: 1.0 for name, _ in blocks}

    def fit(self, X, y=None):
        for _, t in self.blocks:
            t.fit(X, y)
        return self

    def transform(self, X):
        parts = []
        for name, t in self.blocks:
            out = t.transform(X)
            w = self.weights.get(name, 1.0)
            parts.append(out * w)
        return hstack([p if issparse(p) else np.asarray(p) for p in parts]).tocsr()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# --------------------------------------------------------------------------
# 7. Configurações de modelo
# --------------------------------------------------------------------------

PIPELINE_CACHE_DIR = Path("joblib_cache")


def get_model_configs(quick: bool, with_minilm: bool):
    configs = {}

    # baseline oficial exigido pelo enunciado
    configs["baseline_lr"] = (
        Pipeline([("tfidf", word_tfidf_original()),
                  ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        {},
    )

    # Grid enxuto de propósito: cada combinação de pesos refaz o fit dos 3
    # TF-IDF (o item mais caro do pipeline). `memory=` abaixo cacheia esse
    # fit em disco, então repetir o mesmo peso em folds diferentes não é
    # recomputado; ainda assim, mantenha esta lista curta.
    weight_grid = [
        {"word_orig": 1.0, "char_orig": 1.0, "word_norm": 1.0, "stats": 1.0},
        {"word_orig": 1.5, "char_orig": 0.7, "word_norm": 1.0, "stats": 1.0},
    ]

    configs["lexical_svm"] = (
        Pipeline(
            [("features", lexical_features()),
             ("clf", LinearSVC(random_state=RANDOM_STATE, dual="auto"))],
            memory=str(PIPELINE_CACHE_DIR),
        ),
        {} if quick else {
            "features__transformer_weights": weight_grid,
            "clf__C": [0.3, 1.0, 3.0],
        },
    )

    configs["lexical_lr"] = (
        Pipeline(
            [("features", lexical_features()),
             ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))],
            memory=str(PIPELINE_CACHE_DIR),
        ),
        {} if quick else {
            "features__transformer_weights": weight_grid,
            "clf__C": [0.3, 1.0, 3.0],
        },
    )

    if with_minilm:
        hybrid_blocks = [
            ("word_orig", word_tfidf_original()),
            ("char_orig", char_tfidf_original()),
            ("word_norm", word_tfidf_normalized()),
            ("stats", stats_pipeline()),
            ("minilm", MiniLMEmbedder()),
        ]
        configs["hybrid_lexical_semantic"] = (
            Pipeline(
                [("features", DenseHStack(hybrid_blocks)),
                 ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))],
                memory=str(PIPELINE_CACHE_DIR),
            ),
            {} if quick else {"clf__C": [0.3, 1.0, 3.0]},
        )

    return configs


def build_voting_ensemble(lexical_estimator, hybrid_estimator, weights=(1, 1)):
    """Soft-voting entre o melhor modelo lexical e o modelo híbrido semântico
    (ambos já ajustados), como sugerido no diagnóstico."""
    return VotingClassifier(
        estimators=[("lexical", lexical_estimator), ("hybrid", hybrid_estimator)],
        voting="soft", weights=list(weights),
    )


# --------------------------------------------------------------------------
# 8. Avaliação: baseline majoritário + validação dupla (Stratified x Group)
# --------------------------------------------------------------------------

def majority_baseline_accuracy(y_train, y_holdout):
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return accuracy_score(y_holdout, dummy.predict(np.zeros((len(y_holdout), 1))))


def run_model_dual_cv(name, pipeline, grid, X, y, groups, n_splits):
    """Ajusta (com grid search opcional) usando StratifiedKFold e depois mede
    a mesma configuração com GroupKFold (grupo = texto normalizado), sem re-
    otimizar hiperparâmetros no Group CV (evita usar o grupo para tunar)."""
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    if grid:
        search = GridSearchCV(pipeline, grid, cv=skf, scoring="accuracy", n_jobs=-1, refit=True)
        search.fit(X, y)
        best_estimator = search.best_estimator_
        strat_acc = search.best_score_
        best_params = search.best_params_
    else:
        strat_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        pipeline.fit(X, y)
        best_estimator = pipeline
        strat_acc = strat_scores.mean()
        best_params = {}

    n_unique_groups = len(np.unique(groups))
    gkf_splits = min(n_splits, n_unique_groups)
    group_acc = np.nan
    if gkf_splits >= 2:
        gkf = GroupKFold(n_splits=gkf_splits)
        group_scores = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            est = clone(best_estimator)
            est.fit(X[tr_idx], y[tr_idx])
            group_scores.append(accuracy_score(y[va_idx], est.predict(X[va_idx])))
        group_acc = float(np.mean(group_scores))

    gap = strat_acc - group_acc if not np.isnan(group_acc) else np.nan
    elapsed = time.time() - t0
    logger.info(
        "[%s] StratifiedKFold acc=%.4f | GroupKFold(template) acc=%.4f | gap=%.4f | "
        "tempo=%.1fs | params=%s",
        name, strat_acc, group_acc, gap, elapsed, best_params,
    )
    return {
        "model": name,
        "stratified_cv_acc": strat_acc,
        "group_cv_acc": group_acc,
        "gap_stratified_minus_group": gap,
        "fit_time_s": elapsed,
        "best_params": json.dumps(best_params, default=str),
        "estimator": best_estimator,
    }


def check_overfitting(estimator, X_train, y_train, X_hold, y_hold, name):
    train_acc = accuracy_score(y_train, estimator.predict(X_train))
    hold_acc = accuracy_score(y_hold, estimator.predict(X_hold))
    gap = train_acc - hold_acc
    flag = " <-- possível overfitting (gap > 0.08)" if gap > 0.08 else ""
    logger.info(
        "[%s] treino_acc=%.4f | holdout_acc=%.4f | gap=%.4f%s", name, train_acc, hold_acc, gap, flag
    )
    return train_acc, hold_acc, gap


# --------------------------------------------------------------------------
# 9. Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.xlsx")
    parser.add_argument("--test", default="test1.xlsx")
    parser.add_argument("--output", default="test1_predicted_v2.xlsx")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--with-minilm", action="store_true",
                         help="inclui o bloco semântico MiniLM e o ensemble (requer sentence-transformers)")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    setup_logging()
    logger.info("Argumentos: %s", vars(args))

    train_df, test_df = load_data(args.train, args.test)
    group_ids = diagnose_duplicates(train_df)

    X_all = train_df["resp_text"].astype(str).values
    y_all = train_df["clarity"].astype(str).values
    logger.info("Distribuição de classes: %s", pd.Series(y_all).value_counts().to_dict())

    X_tr, X_hold, y_tr, y_hold, groups_tr, _ = train_test_split(
        X_all, y_all, group_ids, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE
    )

    if args.predict_only:
        logger.info("Modo --predict-only: carregando best_model_v2.joblib")
        best_estimator = joblib.load("best_model_v2.joblib")
    else:
        maj_acc = majority_baseline_accuracy(y_tr, y_hold)
        logger.info("Baseline classe majoritária (holdout): acc=%.4f", maj_acc)

        use_minilm = args.with_minilm and minilm_available()
        configs = get_model_configs(quick=not args.full, with_minilm=use_minilm)
        if args.models:
            configs = {k: v for k, v in configs.items() if k in args.models}

        results, estimators = [], {}
        for name, (pipeline, grid) in configs.items():
            try:
                res = run_model_dual_cv(name, pipeline, grid, X_tr, y_tr, groups_tr, args.n_splits)
                results.append(res)
                estimators[name] = res["estimator"]
            except Exception:
                logger.exception("[%s] FALHOU; continuando", name)

        if not results:
            raise RuntimeError("Nenhum modelo terminou com sucesso.")

        results_df = pd.DataFrame(results).sort_values("group_cv_acc", ascending=False)
        results_df.drop(columns=["estimator"]).to_csv("cv_results_v2.csv", index=False)
        logger.info(
            "=== RESULTADOS (ordenado por GroupKFold, o critério de seleção) ===\n%s",
            results_df.drop(columns=["estimator"]).to_string(index=False),
        )

        best_name = results_df.iloc[0]["model"]
        best_estimator = estimators[best_name]

        # ensemble lexical + semântico, se o híbrido estiver disponível e não
        # tiver sido explicitamente excluído por --models
        if use_minilm and "hybrid_lexical_semantic" in estimators:
            lexical_candidates = [n for n in ("lexical_lr", "lexical_svm") if n in estimators]
            if lexical_candidates:
                lexical_name = results_df[results_df["model"].isin(lexical_candidates)] \
                    .sort_values("group_cv_acc", ascending=False).iloc[0]["model"]
                ensemble = build_voting_ensemble(
                    estimators[lexical_name], estimators["hybrid_lexical_semantic"]
                )
                try:
                    res = run_model_dual_cv(
                        "ensemble_soft_voting", ensemble, {}, X_tr, y_tr, groups_tr, args.n_splits
                    )
                    results.append(res)
                    estimators["ensemble_soft_voting"] = res["estimator"]
                    results_df = pd.DataFrame(results).sort_values("group_cv_acc", ascending=False)
                    results_df.drop(columns=["estimator"]).to_csv("cv_results_v2.csv", index=False)
                    best_name = results_df.iloc[0]["model"]
                    best_estimator = estimators[best_name]
                except Exception:
                    logger.exception("[ensemble_soft_voting] FALHOU; mantendo melhor modelo anterior")

        best_row = results_df[results_df["model"] == best_name].iloc[0]
        logger.info(
            "Melhor modelo (critério GroupKFold): %s | group_acc=%.4f | stratified_acc=%.4f",
            best_name, best_row["group_cv_acc"], best_row["stratified_cv_acc"],
        )
        if best_row["group_cv_acc"] < maj_acc:
            logger.warning(
                "ATENÇÃO: acurácia GroupKFold do melhor modelo (%.4f) está abaixo do "
                "baseline majoritário (%.4f) -- revisar antes de entregar.",
                best_row["group_cv_acc"], maj_acc,
            )

        logger.info("=== CHECAGEM DE OVERFITTING (holdout, mesma divisão usada acima) ===")
        check_overfitting(best_estimator, X_tr, y_tr, X_hold, y_hold, best_name)

        logger.info("Retreinando o melhor modelo com 100%% dos dados de treino...")
        best_estimator = clone(best_estimator)
        best_estimator.fit(X_all, y_all)
        joblib.dump(best_estimator, "best_model_v2.joblib")
        logger.info("Modelo salvo em best_model_v2.joblib")

    logger.info("Predizendo conjunto de teste: %d exemplos", len(test_df))
    X_test = test_df["resp_text"].astype(str).values
    preds = best_estimator.predict(X_test)

    out_df = test_df.copy()
    out_df["clarity"] = preds
    out_df.to_excel(args.output, index=False)
    logger.info("Planilha rotulada salva em: %s | distribuição prevista: %s",
                args.output, pd.Series(preds).value_counts().to_dict())
    logger.info("Execução concluída com sucesso.")


if __name__ == "__main__":
    main()