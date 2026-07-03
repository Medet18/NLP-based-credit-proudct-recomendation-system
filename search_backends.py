# -*- coding: utf-8 -*-
"""
search_backends.py — переключаемые методы поиска для экспериментов (Шаг 14).

Один интерфейс, пять конфигураций. Каждый бэкенд принимает каталог и запрос,
возвращает ранжированный список (product_id, score). Жёсткий фильтр по
сумме/сроку/категории применяется поверх ранжирования отдельно (в eval.py),
чтобы честно сравнивать именно качество ранжирования.

Методы:
  1. tfidf   — лексический поиск TF-IDF
  2. bm25    — лексический поиск BM25
  3. dense   — плотные эмбеддинги (paraphrase-multilingual-MiniLM)
  4. hybrid  — комбинация dense + bm25 (взвешенная сумма нормированных оценок)
  5. rerank  — hybrid достаёт топ-N кандидатов, затем эвристический reranker

Про reranking: в промысле здесь предполагался LLM. Для воспроизводимости
эксперимента (стабильные цифры, отсутствие зависимости от API-ключа на защите)
реализован детерминированный эвристический reranker — он подмешивает совпадение
по ключевым атрибутам. Если нужен именно LLM-reranker, его легко подключить
в функцию _heuristic_rerank, не меняя остальной код.
"""

import re
import json
import numpy as np

ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# обогащённый текст продукта (Шаг 9.1): название + назначение + выгоды + описание
def enrich_text(p, enriched=True):
    if enriched:
        return f"{p['name']} {p['category']} {p['purpose']} {p.get('benefits','')} {p['description_text']}"
    return p["description_text"]


def _tokenize(text):
    # простая токенизация под русский/казахский: слова из букв и цифр
    return re.findall(r"[\wа-яёәғқңөұүһі]+", text.lower())


class SearchEngine:
    """Единый движок с переключаемым методом ранжирования."""

    def __init__(self, catalog, method="dense", enriched=True, st_model=None):
        self.catalog = catalog
        self.method = method
        self.enriched = enriched
        self.ids = [p["id"] for p in catalog]
        self.texts = [enrich_text(p, enriched) for p in catalog]

        self._tfidf = None
        self._tfidf_matrix = None
        self._bm25 = None
        self._dense = None          # матрица эмбеддингов продуктов
        self._st_model = st_model   # можно передать готовую модель (чтобы не грузить дважды)

        self._build()

    # --- построение индексов под выбранный метод ------------------------------
    def _build(self):
        need_tfidf = self.method == "tfidf"
        need_bm25 = self.method in ("bm25", "hybrid", "rerank")
        need_dense = self.method in ("dense", "hybrid", "rerank")

        if need_tfidf:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer()
            self._tfidf_matrix = self._tfidf.fit_transform(self.texts)

        if need_bm25:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        if need_dense:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(ST_MODEL_NAME)
            emb = self._st_model.encode(self.texts, normalize_embeddings=True)
            self._dense = np.asarray(emb)

    # --- сырые оценки по каждому методу ---------------------------------------
    def _scores_tfidf(self, query):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self._tfidf.transform([query])
        return cosine_similarity(qv, self._tfidf_matrix).ravel()

    def _scores_bm25(self, query):
        return np.asarray(self._bm25.get_scores(_tokenize(query)))

    def _scores_dense(self, query):
        qv = self._st_model.encode([query], normalize_embeddings=True)
        return (self._dense @ np.asarray(qv).T).ravel()

    @staticmethod
    def _normalize(scores):
        # минимакс-нормализация в [0,1], чтобы складывать разные шкалы
        s = np.asarray(scores, dtype=float)
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return np.zeros_like(s)
        return (s - lo) / (hi - lo)

    # --- эвристический reranker -----------------------------------------------
    def _heuristic_rerank(self, query, candidate_idx, base_scores):
        """Подмешивает бонус за дословное совпадение ключевых терминов.

        Это детерминированная замена LLM-reranker: повышает кандидатов, в тексте
        которых встречаются значимые слова запроса (банк, 'кэшбэк', 'без залога' и т.п.).
        """
        q_tokens = set(_tokenize(query))
        reranked = []
        for i in candidate_idx:
            prod_tokens = set(_tokenize(self.texts[i]))
            overlap = len(q_tokens & prod_tokens)
            bonus = 0.05 * overlap  # небольшой вес, чтобы не перебить семантику
            reranked.append((i, base_scores[i] + bonus))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # --- основной метод: ранжирование -----------------------------------------
    def rank(self, query, top_k=None):
        """Возвращает список (product_id, score), отсортированный по убыванию."""
        if self.method == "tfidf":
            scores = self._scores_tfidf(query)
        elif self.method == "bm25":
            scores = self._scores_bm25(query)
        elif self.method == "dense":
            scores = self._scores_dense(query)
        elif self.method == "hybrid":
            d = self._normalize(self._scores_dense(query))
            b = self._normalize(self._scores_bm25(query))
            scores = 0.6 * d + 0.4 * b   # вес семантики выше
        elif self.method == "rerank":
            d = self._normalize(self._scores_dense(query))
            b = self._normalize(self._scores_bm25(query))
            scores = 0.6 * d + 0.4 * b
            # берём топ-10 кандидатов и реранкируем
            top_n = min(10, len(self.ids))
            cand = list(np.argsort(scores)[::-1][:top_n])
            reranked = self._heuristic_rerank(query, cand, scores)
            ordered = [(self.ids[i], float(s)) for i, s in reranked]
            return ordered[:top_k] if top_k else ordered
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        order = np.argsort(scores)[::-1]
        result = [(self.ids[i], float(scores[i])) for i in order]
        return result[:top_k] if top_k else result


# список методов для экспериментов
ALL_METHODS = ["tfidf", "bm25", "dense", "hybrid", "rerank"]


if __name__ == "__main__":
    catalog = json.load(open("catalog.json", encoding="utf-8"))
    # быстрый тест tfidf (не грузит модель)
    eng = SearchEngine(catalog, method="tfidf")
    print("Метод tfidf, запрос 'хочу машину за 5 млн':")
    for pid, sc in eng.rank("хочу машину за 5 млн", top_k=3):
        print(f"  {pid}  {sc:.3f}")
