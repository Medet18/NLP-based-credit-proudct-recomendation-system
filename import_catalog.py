# -*- coding: utf-8 -*-
"""
import_catalog.py — разовый скрипт: заливает catalog.json в catalog.db
и считает эмбеддинги один раз (сохраняет в таблицу embeddings).

Запуск:  python import_catalog.py
Повторный запуск перезаписывает данные (продукты и эмбеддинги обновляются).
"""

import json
import numpy as np

import db

ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def enrich_text(p):
    """Обогащённый текст для эмбеддинга: название + категория + назначение + выгоды + описание."""
    return f"{p['name']} {p['category']} {p.get('purpose','')} {p.get('benefits','')} {p['description_text']}"


def main():
    print("Инициализация базы...")
    db.init_db()

    print("Чтение catalog.json...")
    catalog = json.load(open("catalog.json", encoding="utf-8"))
    print(f"  продуктов: {len(catalog)}")

    print("Запись продуктов в базу...")
    for p in catalog:
        db.upsert_product(p)

    print("Загрузка модели эмбеддингов (может скачиваться при первом запуске)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ST_MODEL_NAME)
    except Exception as e:
        print(f"  [!] Модель эмбеддингов недоступна ({e.__class__.__name__}).")
        print("      Продукты записаны, но эмбеддинги не посчитаны.")
        print("      recommender откатится на TF-IDF. Запусти снова с интернетом для эмбеддингов.")
        _report()
        return

    print("Расчёт и сохранение эмбеддингов...")
    texts = [enrich_text(p) for p in catalog]
    vectors = model.encode(texts, normalize_embeddings=True)
    for p, vec in zip(catalog, vectors):
        db.save_embedding(p["id"], vec, ST_MODEL_NAME)
    print(f"  сохранено эмбеддингов: {len(catalog)}")

    _report()


def _report():
    print("\nТаблицы в базе:", db.table_list())
    prods = db.get_all_products()
    print(f"Продуктов в базе: {len(prods)}")
    ids, mat = db.get_all_embeddings()
    print(f"Эмбеддингов в базе: {len(ids)}" + (f", размерность {mat.shape[1]}" if mat is not None else ""))
    print("Готово.")


if __name__ == "__main__":
    main()
