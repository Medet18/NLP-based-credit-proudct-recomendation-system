# -*- coding: utf-8 -*-
"""
eval_multilang.py — эксперимент по мультиязычности и code-switching (Блок 5).

Прогоняет три языковых режима (ru / kz / mixed) через несколько мультиязычных
моделей эмбеддингов, считает Top-1 / Top-3 / MRR отдельно по каждому режиму.
Итог — таблица «модель × язык × метрика» + анализ ошибок.

Гипотеза: смешанные (code-switched) запросы дают просадку качества относительно
одноязычных; величина просадки зависит от модели.

ВАЖНО: казахская и смешанная части тест-набора — ЧЕРНОВИК, не выверены носителем
языка. Результаты предварительные; до валидации носителем их нельзя выдавать за
окончательные.

Запуск:  python eval_multilang.py
Результаты: results/multilang_comparison.csv, results/multilang_comparison.png,
            results/codeswitch_errors.csv
"""

import os
import csv
import json
import numpy as np

RESULTS_DIR = "results"

# модели для сравнения: имя -> идентификатор на HuggingFace
MODELS = {
    "MiniLM-multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE": "sentence-transformers/LaBSE",
}
LANGS = ["ru", "kz"]


def enrich_text(p):
    return f"{p['name']} {p['category']} {p.get('purpose','')} {p.get('benefits','')} {p['description_text']}"


def load_data():
    catalog = json.load(open("catalog.json", encoding="utf-8"))
    testset = json.load(open("testset_multilang.json", encoding="utf-8"))["items"]
    return catalog, testset


def metrics_for_lang(model, catalog, testset, lang, prod_matrix, prod_ids):
    """Считает Top-1/Top-3/MRR для одного языкового режима + собирает ошибки."""
    queries = [it[lang] for it in testset]
    q_emb = model.encode(queries, normalize_embeddings=True)
    q_emb = np.asarray(q_emb)

    n = len(testset)
    top1 = top3 = 0
    rr_sum = 0.0
    errors = []

    for i, it in enumerate(testset):
        sims = (prod_matrix @ q_emb[i].T).ravel()
        order = np.argsort(sims)[::-1]
        ranked_ids = [prod_ids[j] for j in order]
        gold = it["gold_product_id"]

        if ranked_ids[0] == gold:
            top1 += 1
        if gold in ranked_ids[:3]:
            top3 += 1
        rank = ranked_ids.index(gold) + 1
        rr_sum += 1.0 / rank

        if ranked_ids[0] != gold:
            errors.append({
                "intent_id": it["intent_id"], "lang": lang, "query": it[lang],
                "gold": gold, "predicted": ranked_ids[0], "gold_rank": rank,
            })

    return {"top1": round(top1 / n, 3), "top3": round(top3 / n, 3),
            "mrr": round(rr_sum / n, 3)}, errors


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    catalog, testset = load_data()
    print(f"Каталог: {len(catalog)} продуктов | Намерений: {len(testset)} (×3 языка = {len(testset)*3} запросов)\n")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[!] sentence-transformers недоступен ({e.__class__.__name__}). Эксперимент невозможен.")
        return

    prod_texts = [enrich_text(p) for p in catalog]
    prod_ids = [p["id"] for p in catalog]

    rows = []
    all_errors = []

    for model_label, model_id in MODELS.items():
        print(f"=== Модель: {model_label} ({model_id}) ===")
        try:
            model = SentenceTransformer(model_id)
        except Exception as e:
            print(f"  [!] не удалось загрузить ({e.__class__.__name__}), пропуск.\n")
            continue

        prod_matrix = np.asarray(model.encode(prod_texts, normalize_embeddings=True))

        for lang in LANGS:
            m, errors = metrics_for_lang(model, catalog, testset, lang, prod_matrix, prod_ids)
            rows.append({"model": model_label, "lang": lang, **m})
            all_errors.extend(errors)
            print(f"  {lang:<6} Top-1={m['top1']:.3f}  Top-3={m['top3']:.3f}  MRR={m['mrr']:.3f}")
        # разница kz относительно ru по Top-1
        ru = next((r for r in rows if r["model"] == model_label and r["lang"] == "ru"), None)
        kz = next((r for r in rows if r["model"] == model_label and r["lang"] == "kz"), None)
        if ru and kz:
            print(f"  -> разница kz vs ru по Top-1: {ru['top1']-kz['top1']:+.3f}")
        print()

    if not rows:
        print("[!] Ни одна модель не загрузилась. Нужен интернет для скачивания моделей.")
        return

    # сохранить таблицу
    with open(os.path.join(RESULTS_DIR, "multilang_comparison.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "lang", "top1", "top3", "mrr"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(RESULTS_DIR, "codeswitch_errors.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent_id", "lang", "query", "gold", "predicted", "gold_rank"])
        w.writeheader()
        w.writerows(all_errors)

    try:
        _plot(rows)
    except Exception as e:
        print(f"[eval_multilang] график не построен ({e.__class__.__name__})")

    print(f"Результаты сохранены в {RESULTS_DIR}/ (multilang_comparison.csv, codeswitch_errors.csv)")
    print("\nНАПОМИНАНИЕ: казахская и смешанная части — черновик, требуется проверка носителем языка.")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(set(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        mr = [r for r in rows if r["model"] == model]
        langs = [r["lang"] for r in mr]
        x = np.arange(len(langs))
        w = 0.25
        ax.bar(x - w, [r["top1"] for r in mr], w, label="Top-1")
        ax.bar(x, [r["top3"] for r in mr], w, label="Top-3")
        ax.bar(x + w, [r["mrr"] for r in mr], w, label="MRR")
        ax.set_xticks(x); ax.set_xticklabels(langs)
        ax.set_ylim(0, 1); ax.set_title(model); ax.legend()
    fig.suptitle("Качество поиска по языковым режимам")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multilang_comparison.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
