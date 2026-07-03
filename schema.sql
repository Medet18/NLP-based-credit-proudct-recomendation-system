-- schema.sql — структура базы каталога финансовых продуктов

-- Продукты: фиксированные поля колонками, категорийно-специфичное — в attributes (JSON)
CREATE TABLE IF NOT EXISTS products (
    id                  TEXT PRIMARY KEY,
    bank                TEXT NOT NULL,
    name                TEXT,
    name_kz             TEXT NOT NULL,
    category            TEXT NOT NULL,
    purpose             TEXT,
    purpose_kz          TEXT,
    currency            TEXT DEFAULT 'KZT',
    min_amount          INTEGER DEFAULT 0,
    max_amount          INTEGER DEFAULT 0,
    term_min_months     INTEGER DEFAULT 0,
    term_max_months     INTEGER DEFAULT 0,
    rate_numeric        REAL DEFAULT 0,
    rate_or_yield       TEXT,
    collateral_required INTEGER DEFAULT 0,
    description_text    TEXT,
    benefits            TEXT,
    fees                TEXT,
    requirements        TEXT,           -- JSON-строка
    attributes          TEXT,           -- JSON-строка
    source_url          TEXT,
    bank_url            TEXT,
    data_source         TEXT DEFAULT 'illustrative',
    updated_at          TEXT
);

-- Эмбеддинги: вектор продукта в виде BLOB + имя модели
CREATE TABLE IF NOT EXISTS embeddings (
    product_id  TEXT PRIMARY KEY,
    vector      BLOB NOT NULL,
    model_name  TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

-- Обратная связь по рекомендациям
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  TEXT,
    query       TEXT,
    rating      INTEGER,                -- +1 / -1
    created_at  TEXT
);

-- Журнал запросов
CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT,
    extracted   TEXT,                   -- JSON: что извлекли
    found       INTEGER,                -- 1 если нашлось решение
    created_at  TEXT
);
