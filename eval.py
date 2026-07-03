# -*- coding: utf-8 -*-
"""
eval.py — оценка качества поиска (Шаги 13, 14, 15, 16, 17).

Запуск:
  python eval.py                  # полный прогон: метрики всех методов + ablation + ошибки
  python eval.py --method dense   # метрики одного метода
  python eval.py --no-filter      # без жёсткого фильтра (чистое ранжирование)

Результаты складываются в папку results/ (CSV + график).
"""

import os
import csv
import json
import argparse

from extract import extract
from search_backends import SearchEngine, ALL_METHODS

RESULTS_DIR = "results"


# --- жёсткий фильтр (та же логика, что в recommender) ------------------------
def passes_filter(product, parsed):
    cat = parsed.get("intent_category")
    amount = parsed.get("amount")
    term = parsed.get("term_months")
    conf = parsed.get("intent_confidence", 0)

    if cat and cat != "не_определено" and conf >= 0.6:
        if product["category"] != cat:
            return False
    if amount and product.get("max_amount", 0) > 0:
        if not (product["min_amount"] <= amount <= product["max_amount"]):
            return False
    if term and product.get("term_max_months", 0) > 0:
        if not (product["term_min_months"] <= term <= product["term_max_months"]):
            return False
    return True


# --- метрики ------------------------------------------------------------------
def evaluate(engine, testset, catalog, use_filter=True, parsed_cache=None):
    """Возвращает (метрики dict, список провалов).

    Метрики: top1, top3, mrr.
    """
    by_id = {p["id"]: p for p in catalog}
    n = len(testset)
    top1_hits = 0
    top3_hits = 0
    rr_sum = 0.0
    failures = []

    for item in testset:
        query = item["query"]
        gold = item["gold_product_id"]
        acceptable = set([gold] + item.get("acceptable_product_ids", []))

        parsed = (parsed_cache or {}).get(item["id"]) or extract(query)

        ranked = engine.rank(query)  # [(pid, score), ...] весь список

        # применяем фильтр поверх ранжирования
        if use_filter:
            filtered = [(pid, sc) for pid, sc in ranked if passes_filter(by_id[pid], parsed)]
            if not filtered:               # фильтр отсёк всё — откатываемся к сырому
                filtered = ranked
            ranked = filtered

        ranked_ids = [pid for pid, _ in ranked]

        # Top-1: правильный (gold) ровно на первом месте
        if ranked_ids and ranked_ids[0] == gold:
            top1_hits += 1

        # Top-3: любой приемлемый в первой тройке
        if acceptable & set(ranked_ids[:3]):
            top3_hits += 1

        # MRR: позиция первого приемлемого
        rr = 0.0
        for pos, pid in enumerate(ranked_ids, start=1):
            if pid in acceptable:
                rr = 1.0 / pos
                break
        rr_sum += rr

        # фиксируем провал Top-1 для анализа ошибок
        if not (ranked_ids and ranked_ids[0] == gold):
            failures.append({
                "id": item["id"], "query": query,
                "gold": gold, "predicted": ranked_ids[0] if ranked_ids else "—",
                "gold_rank": (ranked_ids.index(gold) + 1) if gold in ranked_ids else -1,
            })

    metrics = {
        "top1": round(top1_hits / n, 3),
        "top3": round(top3_hits / n, 3),
        "mrr": round(rr_sum / n, 3),
    }
    return metrics, failures


# --- общий загрузчик ----------------------------------------------------------
def load_data():
    catalog = json.load(open("catalog.json", encoding="utf-8"))
    testset = json.load(open("testset.json", encoding="utf-8"))["items"]
    return catalog, testset


def shared_st_model():
    """Грузим модель эмбеддингов один раз и переиспользуем во всех методах."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"[eval] модель эмбеддингов недоступна ({e.__class__.__name__}). "
              f"Методы dense/hybrid/rerank будут пропущены.")
        return None


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Шаг 14: сравнение методов ------------------------------------------------
def run_comparison(catalog, testset, use_filter=True):
    ensure_results_dir()
    st_model = shared_st_model()

    # предрасчёт извлечения параметров (одинаков для всех методов)
    parsed_cache = {it["id"]: extract(it["query"]) for it in testset}

    rows = []
    methods = ALL_METHODS if st_model else ["tfidf", "bm25"]
    for method in methods:
        eng = SearchEngine(catalog, method=method, enriched=True, st_model=st_model)
        m, _ = evaluate(eng, testset, catalog, use_filter=use_filter, parsed_cache=parsed_cache)
        rows.append({"method": method, **m})
        print(f"  {method:<8} Top-1={m['top1']:.3f}  Top-3={m['top3']:.3f}  MRR={m['mrr']:.3f}")

    # сохранить CSV
    path = os.path.join(RESULTS_DIR, "comparison.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "top1", "top3", "mrr"])
        w.writeheader()
        w.writerows(rows)

    # график
    try:
        _plot_comparison(rows)
    except Exception as e:
        print(f"[eval] график не построен ({e.__class__.__name__})")

    return rows


def _plot_comparison(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [r["method"] for r in rows]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, [r["top1"] for r in rows], w, label="Top-1")
    ax.bar(x, [r["top3"] for r in rows], w, label="Top-3")
    ax.bar(x + w, [r["mrr"] for r in rows], w, label="MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение методов поиска")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=120)
    plt.close(fig)


# --- Шаг 15: ablation study ---------------------------------------------------
def run_ablation(catalog, testset):
    """Вклад компонентов: базовый dense -> +обогащённый текст -> +гибрид -> +rerank
    плюс эффект жёсткого фильтра."""
    ensure_results_dir()
    st_model = shared_st_model()
    if not st_model:
        print("[eval] ablation пропущен: нужна модель эмбеддингов.")
        return []

    parsed_cache = {it["id"]: extract(it["query"]) for it in testset}
    configs = [
        ("dense, без обогащения, без фильтра",
         dict(method="dense", enriched=False), False),
        ("dense, обогащённый текст, без фильтра",
         dict(method="dense", enriched=True), False),
        ("dense, обогащённый текст, + фильтр",
         dict(method="dense", enriched=True), True),
        ("hybrid (dense+bm25), + фильтр",
         dict(method="hybrid", enriched=True), True),
        ("hybrid + rerank, + фильтр",
         dict(method="rerank", enriched=True), True),
    ]

    rows = []
    for label, eng_kwargs, use_filter in configs:
        eng = SearchEngine(catalog, st_model=st_model, **eng_kwargs)
        m, _ = evaluate(eng, testset, catalog, use_filter=use_filter, parsed_cache=parsed_cache)
        rows.append({"config": label, **m})
        print(f"  {label:<42} Top-1={m['top1']:.3f}  Top-3={m['top3']:.3f}  MRR={m['mrr']:.3f}")

    path = os.path.join(RESULTS_DIR, "ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "top1", "top3", "mrr"])
        w.writeheader()
        w.writerows(rows)
    return rows


# --- Шаг 16: анализ ошибок ----------------------------------------------------
def run_error_analysis(catalog, testset, method="hybrid"):
    ensure_results_dir()
    st_model = shared_st_model()
    if not st_model and method in ("dense", "hybrid", "rerank"):
        method = "bm25"
    parsed_cache = {it["id"]: extract(it["query"]) for it in testset}
    by_id = {p["id"]: p for p in catalog}

    eng = SearchEngine(catalog, method=method, enriched=True, st_model=st_model)
    _, failures = evaluate(eng, testset, catalog, use_filter=True, parsed_cache=parsed_cache)

    # категоризация ошибки
    rows = []
    for f in failures:
        parsed = parsed_cache[f["id"]]
        gold = by_id[f["gold"]]
        pred = by_id.get(f["predicted"])
        if pred and pred["category"] != gold["category"]:
            kind = "не та категория"
        elif f["gold_rank"] == -1:
            kind = "gold отфильтрован (сумма/срок)"
        elif f["gold_rank"] <= 3:
            kind = "семантический промах (gold близко, в топ-3)"
        else:
            kind = "семантический промах (gold далеко)"
        rows.append({
            "id": f["id"], "query": f["query"], "gold": f["gold"],
            "predicted": f["predicted"], "gold_rank": f["gold_rank"], "kind": kind,
        })

    path = os.path.join(RESULTS_DIR, "error_analysis.csv")
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "query", "gold", "predicted", "gold_rank", "kind"])
        w.writeheader()
        w.writerows(rows)

    print(f"  Метод: {method}. Ошибок Top-1: {len(rows)} из {len(testset)}")
    from collections import Counter
    for kind, cnt in Counter(r["kind"] for r in rows).items():
        print(f"    {kind}: {cnt}")
    return rows


# --- Шаг 17: покрытие объяснением ---------------------------------------------
def run_explanation_coverage(catalog, testset):
    """Доля совпавших критериев (цель/сумма/срок), которые упомянуты в объяснении."""
    ensure_results_dir()
    from recommender import Recommender
    rec = Recommender()  # использует свой бэкенд (st или tfidf)
    parsed_cache = {it["id"]: extract(it["query"]) for it in testset}

    coverages = []
    for item in testset:
        parsed = parsed_cache[item["id"]]
        results = rec.recommend(parsed, top_k=1)
        if not results:
            continue
        r = results[0]
        expl = r["explanation"].lower()
        # какие критерии заданы в запросе
        criteria = []
        if parsed["intent_category"] != "не_определено":
            criteria.append("цель")
        if parsed["amount"]:
            criteria.append("сумма")
        if parsed["term_months"]:
            criteria.append("срок")
        if not criteria:
            continue
        # сколько из них упомянуто в объяснении
        covered = 0
        if "цел" in expl: covered += ("цель" in criteria)
        if "сумм" in expl or "лимит" in expl: covered += ("сумма" in criteria)
        if "срок" in expl or "мес" in expl: covered += ("срок" in criteria)
        coverages.append(covered / len(criteria))

    avg = round(sum(coverages) / len(coverages), 3) if coverages else 0.0
    path = os.path.join(RESULTS_DIR, "explanation_coverage.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["avg_coverage", "n_queries"])
        w.writerow([avg, len(coverages)])
    print(f"  Среднее покрытие объяснением: {avg} (по {len(coverages)} запросам)")
    return avg


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", help="прогнать один метод", choices=ALL_METHODS)
    ap.add_argument("--no-filter", action="store_true", help="без жёсткого фильтра")
    args = ap.parse_args()

    catalog, testset = load_data()
    print(f"Каталог: {len(catalog)} продуктов | Тест-набор: {len(testset)} запросов\n")

    if args.method:
        st_model = shared_st_model() if args.method in ("dense", "hybrid", "rerank") else None
        eng = SearchEngine(catalog, method=args.method, st_model=st_model)
        parsed_cache = {it["id"]: extract(it["query"]) for it in testset}
        m, _ = evaluate(eng, testset, catalog, use_filter=not args.no_filter, parsed_cache=parsed_cache)
        print(f"Метод {args.method}: Top-1={m['top1']}  Top-3={m['top3']}  MRR={m['mrr']}")
        return

    print("=== Шаг 14: Сравнение методов поиска ===")
    run_comparison(catalog, testset, use_filter=not args.no_filter)
    print("\n=== Шаг 15: Ablation study ===")
    run_ablation(catalog, testset)
    print("\n=== Шаг 16: Анализ ошибок ===")
    run_error_analysis(catalog, testset)
    print("\n=== Шаг 17: Покрытие объяснением ===")
    run_explanation_coverage(catalog, testset)
    print(f"\nВсе результаты сохранены в папку {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
