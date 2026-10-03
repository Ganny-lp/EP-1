import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ["c1", "c234", "c5"]
OUT = Path("data_audit")
LOG = OUT / "audit.log"
logger = logging.getLogger("ep1_data_audit")

WS = re.compile(r"\s+")
CTRL = re.compile(r"[\x80-\x9f]")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b", re.I)
UPPER = re.compile(r"\b[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ]{2,}\b")
WORD = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)


def setup():
    OUT.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def clean(x):
    if pd.isna(x):
        return ""
    return WS.sub(" ", CTRL.sub(" ", str(x))).strip()


def norm(x):
    return clean(x).casefold()


def load(train_path, test_path):
    tr = pd.read_excel(train_path)
    te = pd.read_excel(test_path)
    if not {"resp_text", "clarity"} <= set(tr.columns):
        raise ValueError(f"Treino precisa ter resp_text e clarity: {list(tr.columns)}")
    if "resp_text" not in te.columns:
        raise ValueError(f"Teste precisa ter resp_text: {list(te.columns)}")
    tr = tr.copy()
    te = te.copy()
    tr["resp_text"] = tr["resp_text"].map(clean)
    te["resp_text"] = te["resp_text"].map(clean)
    tr["clarity"] = tr["clarity"].astype(str).str.strip().str.lower()
    bad = sorted(set(tr.clarity) - set(LABELS))
    if bad:
        raise ValueError(f"Rótulos inesperados: {bad}")
    logger.info("Treino=%d | teste=%d", len(tr), len(te))
    logger.info("Distribuição=%s", tr.clarity.value_counts().to_dict())
    return tr, te


def features(text):
    text = clean(text)
    words = WORD.findall(text)
    n = len(words)
    s = len([x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()])
    chars = len(text)
    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    uniq = len(set(w.casefold() for w in words))
    punct = sum(not c.isalnum() and not c.isspace() for c in text)
    return {
        "chars": chars, "words": n, "unique_words": uniq, "sentences": s,
        "paragraphs": len([x for x in text.split("\n\n") if x.strip()]),
        "avg_word_len": sum(map(len, words))/n if n else 0,
        "avg_sentence_words": n/s if s else 0,
        "type_token_ratio": uniq/n if n else 0,
        "alpha_ratio": alpha/chars if chars else 0,
        "digit_ratio": digits/chars if chars else 0,
        "uppercase_ratio": upper/alpha if alpha else 0,
        "punctuation_ratio": punct/chars if chars else 0,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "colon_count": text.count(":"),
        "semicolon_count": text.count(";"),
        "comma_count": text.count(","),
        "parentheses_count": text.count("(")+text.count(")"),
        "quote_count": text.count('"')+text.count("“")+text.count("”"),
        "url_count": len(URL.findall(text)),
        "email_count": len(EMAIL.findall(text)),
        "number_count": len(NUMBER.findall(text)),
        "date_count": len(DATE.findall(text)),
        "uppercase_token_count": len(UPPER.findall(text)),
        "has_url": int(bool(URL.search(text))),
        "has_email": int(bool(EMAIL.search(text))),
        "has_number": int(bool(NUMBER.search(text))),
        "has_date": int(bool(DATE.search(text))),
        "has_list_marker": int(bool(re.search(r"(?:^|\s)[•\-–—]\s+", text))),
    }


def audit_duplicates(tr):
    x = tr[["resp_text", "clarity"]].copy()
    x["normalized_text"] = x.resp_text.map(norm)
    g = x.groupby("normalized_text").agg(
        occurrences=("clarity", "size"),
        n_labels=("clarity", "nunique"),
        labels=("clarity", lambda s: "|".join(sorted(set(s))))
    ).reset_index()
    g = g[g.occurrences > 1].copy()
    g["conflict"] = g.n_labels > 1
    g = g.sort_values(["conflict", "occurrences"], ascending=[False, False])
    g.to_csv(OUT/"exact_duplicate_groups.csv", index=False)
    conflicts = g[g.conflict]
    conflicts.to_csv(OUT/"conflicting_texts.csv", index=False)

    repeated = int(g.occurrences.sum())
    conflict_rows = int(conflicts.occurrences.sum())
    correct = 0
    for key in g.normalized_text:
        vc = x.loc[x.normalized_text == key, "clarity"].value_counts()
        correct += int(vc.max())
    ceiling = correct/repeated if repeated else np.nan

    logger.info("Grupos duplicados=%d | linhas duplicadas=%d", len(g), repeated)
    logger.info("Grupos conflitantes=%d | linhas conflitantes=%d",
                len(conflicts), conflict_rows)
    logger.info("Teto majoritário nos textos repetidos=%.4f", ceiling)
    return len(g), repeated, len(conflicts), conflict_rows, ceiling


def distinctive_terms(tr):
    logger.info("Calculando termos distintivos...")
    v = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=.98,
                        sublinear_tf=True, max_features=100000)
    X = v.fit_transform(tr.resp_text)
    terms = np.asarray(v.get_feature_names_out())
    rows = []
    y = tr.clarity.values
    for label in LABELS:
        a = np.asarray(X[y == label].mean(0)).ravel()
        b = np.asarray(X[y != label].mean(0)).ravel()
        score = np.log((a+1e-6)/(b+1e-6))
        for rank, i in enumerate(np.argsort(score)[::-1][:150], 1):
            rows.append({"label": label, "rank": rank, "term": terms[i],
                         "score": float(score[i]),
                         "class_mean_tfidf": float(a[i]),
                         "other_mean_tfidf": float(b[i])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT/"class_distinctive_terms.csv", index=False)
    return out


def near_duplicates(tr, max_rows=6000, threshold=.92):
    logger.info("Quase-duplicatas: amostra máxima=%d", max_rows)
    if len(tr) > max_rows:
        parts = []
        per = max_rows // len(LABELS)
        for lab in LABELS:
            s = tr[tr.clarity == lab]
            parts.append(s.sample(min(per, len(s)), random_state=42))
        tr = pd.concat(parts).sample(max_rows, random_state=42).reset_index(drop=True)

    v = TfidfVectorizer(analyzer="char", ngram_range=(3,5),
                        min_df=2, max_features=80000, sublinear_tf=True)
    X = v.fit_transform(tr.resp_text)
    rows = []
    block = 400
    for start in range(0, len(tr), block):
        end = min(start+block, len(tr))
        sim = cosine_similarity(X[start:end], X)
        for li in range(end-start):
            i = start+li
            js = np.where(sim[li, i+1:] >= threshold)[0]
            for off in js:
                j = i+1+int(off)
                rows.append({
                    "row_i": i, "row_j": j, "similarity": float(sim[li,j]),
                    "label_i": tr.iloc[i].clarity, "label_j": tr.iloc[j].clarity,
                    "same_label": int(tr.iloc[i].clarity == tr.iloc[j].clarity),
                    "text_i": tr.iloc[i].resp_text[:1000],
                    "text_j": tr.iloc[j].resp_text[:1000],
                })
        logger.info("Quase-duplicatas: %.1f%%", 100*end/len(tr))
    out = pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame()
    out.to_csv(OUT/"near_duplicate_pairs.csv", index=False)
    logger.info("Pares >= %.2f: %d", threshold, len(out))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.xlsx")
    p.add_argument("--test", default="test1.xlsx")
    p.add_argument("--near-duplicates", action="store_true")
    args = p.parse_args()
    setup()
    t0 = time.time()
    tr, te = load(args.train, args.test)

    logger.info("Extraindo features estruturais...")
    trf = pd.DataFrame([features(x) for x in tr.resp_text])
    tef = pd.DataFrame([features(x) for x in te.resp_text])
    trf.insert(0, "row_id", np.arange(len(tr)))
    tef.insert(0, "row_id", np.arange(len(te)))
    trf["clarity"] = tr.clarity.values
    trf.to_csv(OUT/"row_level_stats.csv", index=False)
    tef.to_csv(OUT/"row_level_stats_test.csv", index=False)
    trf.drop(columns="clarity").to_csv(OUT/"engineered_features_train.csv", index=False)
    tef.to_csv(OUT/"engineered_features_test.csv", index=False)

    dup = audit_duplicates(tr)
    terms = distinctive_terms(tr)

    # Resumo por classe.
    numeric = [c for c in trf.columns if c not in ("row_id","clarity")]
    summary = trf.groupby("clarity")[numeric].agg(["mean","median","std"]).round(4)
    summary.to_csv(OUT/"class_text_stats.csv")

    if args.near_duplicates:
        near_duplicates(tr)

    counts = tr.clarity.value_counts()
    lines = [
        "ACH2118 EP1 - AUDITORIA DOS DADOS", "="*70,
        f"Treino: {len(tr)}", f"Teste: {len(te)}", "",
        "DISTRIBUIÇÃO"
    ]
    for lab in LABELS:
        lines.append(f"{lab}: {counts.get(lab,0)} ({counts.get(lab,0)/len(tr):.2%})")
    lines += [
        "", "DUPLICATAS",
        f"grupos_duplicados: {dup[0]}",
        f"linhas_duplicadas: {dup[1]}",
        f"grupos_conflitantes: {dup[2]}",
        f"linhas_conflitantes: {dup[3]}",
        f"teto_majoritario_repetidos: {dup[4]:.4%}",
        "", "PRÓXIMOS EXPERIMENTOS",
        "1. Word TF-IDF + char TF-IDF.",
        "2. Adicionar features estruturais.",
        "3. Testar pesos word/char/estruturais.",
        "4. LinearSVC e LogisticRegression.",
        "5. Combinar com embeddings MiniLM já cacheados.",
        "6. Ensemble lexical + semântico.",
        "7. Avaliar split por grupos para evitar vazamento entre duplicatas.",
        "", "ARQUIVOS GERADOS:",
        *[f"- {x.name}" for x in OUT.glob("*.csv")]
    ]
    (OUT/"audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Concluído em %.1fs | saída=%s", time.time()-t0, OUT.resolve())


if __name__ == "__main__":
    main()
