# -*- coding: utf-8 -*-
"""
db.py — слой доступа к базе SQLite (catalog.db).

Хранит продукты, эмбеддинги (BLOB), фидбэк и журнал запросов.
Категорийно-специфичные поля продукта лежат в колонке attributes как JSON-строка,
поэтому схема не ломается при новых типах продуктов.
"""

import os
import json
import sqlite3
from datetime import datetime

import numpy as np

DB_PATH = os.environ.get("CATALOG_DB", "catalog.db")
SCHEMA_PATH = "schema.sql"

# поля, которые лежат отдельными колонками (остальное уходит в attributes)
PRODUCT_COLUMNS = [
    "id", "bank", "name", "name_kz", "category", "purpose", "purpose_kz", "currency",
    "min_amount", "max_amount", "term_min_months", "term_max_months",
    "rate_numeric", "rate_or_yield", "collateral_required",
    "description_text", "benefits", "fees", "requirements", "attributes",
    "source_url", "bank_url", "data_source", "updated_at",
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Создаёт таблицы по schema.sql, если их нет."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = f.read()
    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()


# --- сериализация продукта <-> строка БД --------------------------------------
def _product_to_row(p):
    """dict продукта -> dict значений под колонки БД."""
    row = {
        "id": p["id"], "bank": p["bank"], "name": p["name"],
        "name_kz": p.get("name_kz", p["name"]),
        "category": p["category"], "purpose": p.get("purpose", ""),
        "purpose_kz": p.get("purpose_kz", ""),
        "currency": p.get("currency", "KZT"),
        "min_amount": p.get("min_amount", 0), "max_amount": p.get("max_amount", 0),
        "term_min_months": p.get("term_min_months", 0),
        "term_max_months": p.get("term_max_months", 0),
        "rate_numeric": p.get("rate_numeric", 0) or 0,
        "rate_or_yield": p.get("rate_or_yield", ""),
        "collateral_required": 1 if p.get("collateral_required") else 0,
        "description_text": p.get("description_text", ""),
        "benefits": p.get("benefits", ""), "fees": p.get("fees", ""),
        "requirements": json.dumps(p.get("requirements", {}), ensure_ascii=False),
        "attributes": json.dumps(p.get("attributes", {}), ensure_ascii=False),
        "source_url": p.get("source_url", ""),
        "bank_url": p.get("bank_url", ""),
        "data_source": p.get("data_source", "illustrative"),
        "updated_at": p.get("updated_at") or datetime.now().isoformat(timespec="seconds"),
    }
    return row


def _row_to_product(row):
    """строка БД -> dict продукта (как в catalog.json)."""
    p = dict(row)
    p["collateral_required"] = bool(p.get("collateral_required"))
    for jf in ("requirements", "attributes"):
        try:
            p[jf] = json.loads(p[jf]) if p[jf] else {}
        except (json.JSONDecodeError, TypeError):
            p[jf] = {}
    return p


# --- продукты -----------------------------------------------------------------
def get_all_products():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM products ORDER BY category, bank").fetchall()
    conn.close()
    return [_row_to_product(r) for r in rows]


def get_product(product_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    conn.close()
    return _row_to_product(row) if row else None


def upsert_product(p):
    """Вставить или обновить продукт. updated_at проставляется автоматически."""
    p = dict(p)
    p["updated_at"] = datetime.now().isoformat(timespec="seconds")
    row = _product_to_row(p)
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "id")
    sql = (f"INSERT INTO products ({','.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT(id) DO UPDATE SET {updates}")
    conn = get_connection()
    conn.execute(sql, [row[c] for c in cols])
    conn.commit()
    conn.close()
    return row["updated_at"]


def delete_product(product_id):
    conn = get_connection()
    conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
    conn.execute("DELETE FROM embeddings WHERE product_id = ?", (product_id,))
    conn.commit()
    conn.close()


# --- эмбеддинги ---------------------------------------------------------------
def save_embedding(product_id, vector, model_name):
    """vector — numpy-массив float32. Хранится как BLOB."""
    vec = np.asarray(vector, dtype=np.float32)
    blob = vec.tobytes()
    conn = get_connection()
    conn.execute(
        "INSERT INTO embeddings (product_id, vector, model_name) VALUES (?,?,?) "
        "ON CONFLICT(product_id) DO UPDATE SET vector=excluded.vector, model_name=excluded.model_name",
        (product_id, blob, model_name),
    )
    conn.commit()
    conn.close()


def get_all_embeddings(model_name=None):
    """Возвращает (list_product_ids, np.array[N,D]) или (ids, None) если пусто."""
    conn = get_connection()
    if model_name:
        rows = conn.execute("SELECT product_id, vector FROM embeddings WHERE model_name = ?",
                            (model_name,)).fetchall()
    else:
        rows = conn.execute("SELECT product_id, vector FROM embeddings").fetchall()
    conn.close()
    if not rows:
        return [], None
    ids = [r["product_id"] for r in rows]
    vecs = [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
    return ids, np.vstack(vecs)


# --- фидбэк -------------------------------------------------------------------
def save_feedback(product_id, query, rating):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (product_id, query, rating, created_at) VALUES (?,?,?,?)",
        (product_id, query, int(rating), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_feedback_stats():
    """Сводка по продуктам: сколько 👍 и 👎."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT product_id, "
        "SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS up, "
        "SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS down "
        "FROM feedback GROUP BY product_id ORDER BY up DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- журнал запросов ----------------------------------------------------------
def log_query(query, extracted, found):
    conn = get_connection()
    conn.execute(
        "INSERT INTO query_log (query, extracted, found, created_at) VALUES (?,?,?,?)",
        (query, json.dumps(extracted, ensure_ascii=False), 1 if found else 0,
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_recent_queries(n=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_query_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) AS c FROM query_log").fetchone()["c"]
    not_found = conn.execute("SELECT COUNT(*) AS c FROM query_log WHERE found = 0").fetchone()["c"]
    conn.close()
    return {"total": total, "not_found": not_found,
            "not_found_rate": round(not_found / total, 3) if total else 0.0}


def table_list():
    conn = get_connection()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    conn.close()
    return [r["name"] for r in rows]
