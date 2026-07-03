# -*- coding: utf-8 -*-
"""
recommender.py — подбор продуктов под запрос клиента.

Этапы:
  1. Загрузка catalog.json, построение векторов по description_text.
  2. Запрос -> вектор -> косинусная близость со всеми продуктами.
  3. Жёсткий фильтр по сумме / сроку / категории.
  4. Топ-3 продукта с оценкой релевантности и объяснением.

Два бэкенда векторизации:
  - 'st'    : sentence-transformers (мультиязычные эмбеддинги) — основной;
  - 'tfidf' : TF-IDF из scikit-learn — запасной, работает офлайн без загрузки моделей.
Выбирается автоматически: если sentence-transformers и модель доступны — берём их,
иначе откатываемся на TF-IDF. Так прототип работает даже без интернета.
"""

import json
import os
import numpy as np

from calculator import annuity_payment, format_kzt

ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Recommender:
    def __init__(self, catalog_path="catalog.json", prefer_backend="auto"):
        # переменная окружения REC_BACKEND переопределяет режим:
        #   REC_BACKEND=tfidf -> принудительно офлайн без загрузки модели
        #   REC_BACKEND=st    -> только эмбеддинги
        prefer_backend = os.environ.get("REC_BACKEND", prefer_backend)

        # каталог: сначала пробуем базу, если её нет — JSON-файл
        self.catalog = self._load_catalog(catalog_path)

        self.texts = [self._product_text(p) for p in self.catalog]
        self.backend = None
        self._st_model = None
        self._tfidf = None
        self._matrix = None
        self._intent_clf = None
        self._bm25 = None

        self._init_backend(prefer_backend)

    # --- загрузка каталога: база или JSON -------------------------------------
    @staticmethod
    def _load_catalog(catalog_path):
        try:
            import db
            if os.path.exists(db.DB_PATH):
                products = db.get_all_products()
                if products:
                    return products
        except Exception:
            pass
        # фолбэк: JSON-файл
        with open(catalog_path, encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        """Перечитать каталог и пересобрать индекс (после правок в админке)."""
        self.catalog = self._load_catalog("catalog.json")
        self.texts = [self._product_text(p) for p in self.catalog]
        self._init_backend(os.environ.get("REC_BACKEND", "auto"))

    # --- текст продукта для векторизации --------------------------------------
    @staticmethod
    def _product_text(p):
        # обогащённый текст: название + категория + назначение + выгоды + описание
        return (f"{p.get('name','')} {p['category']} {p.get('purpose','')} "
                f"{p.get('benefits','')} {p['description_text']}")

    # --- инициализация бэкенда ------------------------------------------------
    def _init_backend(self, prefer):
        if prefer in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(ST_MODEL_NAME)
                # пробуем взять предпосчитанные эмбеддинги из базы
                matrix = self._load_embeddings_from_db()
                if matrix is not None:
                    self._matrix = matrix
                else:
                    emb = self._st_model.encode(self.texts, normalize_embeddings=True)
                    self._matrix = np.asarray(emb)
                self.backend = "st"
                # BM25-индекс для гибридного поиска (лексическая составляющая)
                try:
                    from rank_bm25 import BM25Okapi
                    self._bm25 = BM25Okapi([self._tokenize(t) for t in self.texts])
                except Exception:
                    self._bm25 = None
                # классификатор намерения на эмбеддингах (та же модель)
                try:
                    from intent_classifier import IntentClassifier
                    self._intent_clf = IntentClassifier(self._st_model)
                except Exception:
                    self._intent_clf = None
                return
            except Exception as e:
                if prefer == "st":
                    raise
                print(f"[recommender] sentence-transformers недоступен ({e.__class__.__name__}), "
                      f"откатываюсь на TF-IDF.")

        # фолбэк: TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer()
        self._matrix = self._tfidf.fit_transform(self.texts)
        self.backend = "tfidf"

    def _load_embeddings_from_db(self):
        """Берёт предпосчитанные эмбеддинги из базы в порядке self.catalog.
        Возвращает матрицу или None, если их нет / не совпадают с каталогом."""
        try:
            import db
            if not os.path.exists(db.DB_PATH):
                return None
            ids, mat = db.get_all_embeddings(ST_MODEL_NAME)
            if mat is None:
                return None
            by_id = {pid: i for i, pid in enumerate(ids)}
            # эмбеддинги должны покрывать весь текущий каталог
            if not all(p["id"] in by_id for p in self.catalog):
                return None
            order = [by_id[p["id"]] for p in self.catalog]
            return mat[order]
        except Exception:
            return None

    # --- вектор запроса -------------------------------------------------------
    def _embed_query(self, query):
        if self.backend == "st":
            v = self._st_model.encode([query], normalize_embeddings=True)
            return np.asarray(v)
        else:
            return self._tfidf.transform([query])

    # --- вспомогательное для гибридного поиска --------------------------------
    @staticmethod
    def _tokenize(text):
        import re
        return re.findall(r"[\wа-яёәғқңөұүһі]+", text.lower())

    @staticmethod
    def _normalize(scores):
        # минимакс-нормализация в [0,1], чтобы складывать разные шкалы
        s = np.asarray(scores, dtype=float)
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return np.zeros_like(s)
        return (s - lo) / (hi - lo)

    def _dense_scores(self, query):
        qv = self._embed_query(query)
        if self.backend == "st":
            return (self._matrix @ qv.T).ravel()
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(qv, self._matrix).ravel()

    def _heuristic_rerank(self, query, scores, top_n=10):
        """Переранжирование топ-N кандидатов: бонус за дословное совпадение
        значимых слов запроса с текстом продукта (детерминированно)."""
        q_tokens = set(self._tokenize(query))
        order = list(np.argsort(scores)[::-1][:top_n])
        reranked = []
        for i in order:
            overlap = len(q_tokens & set(self._tokenize(self.texts[i])))
            reranked.append((i, scores[i] + 0.05 * overlap))
        reranked.sort(key=lambda x: x[1], reverse=True)
        # собираем полный вектор оценок: реранкированные сверху, остальные ниже
        new_scores = np.array(scores, dtype=float)
        for rank_pos, (i, s) in enumerate(reranked):
            new_scores[i] = s
        return new_scores

    # --- основное ранжирование: гибрид dense + BM25 + reranking ---------------
    def _similarities(self, query):
        """Итоговая конфигурация системы — гибридный поиск с переранжированием.
        Объединяет семантическую (dense) и лексическую (BM25) близость, затем
        применяет детерминированное переранжирование. На TF-IDF-фолбэке и при
        отсутствии BM25 используется чистая косинусная близость."""
        dense = self._dense_scores(query)

        # если BM25 доступен (режим st) — строим гибрид; иначе только dense
        if self._bm25 is not None:
            d = self._normalize(dense)
            b = self._normalize(np.asarray(self._bm25.get_scores(self._tokenize(query))))
            scores = 0.6 * d + 0.4 * b           # вес семантики выше лексики
            scores = self._heuristic_rerank(query, scores)
            return scores
        return dense

    # --- жёсткий фильтр -------------------------------------------------------
    @staticmethod
    def _passes_filter(product, parsed):
        cat = parsed.get("intent_category")
        amount = parsed.get("amount")
        term = parsed.get("term_months")
        conf = parsed.get("intent_confidence", 0)

        # 1) категория: отсекаем по категории только если уверены (conf >= 0.6)
        if cat and cat != "не_определено" and conf >= 0.6:
            if product["category"] != cat:
                return False

        # 2) сумма: проверяем для продуктов с заданными лимитами (>0)
        if amount and product.get("max_amount", 0) > 0:
            if not (product["min_amount"] <= amount <= product["max_amount"]):
                return False

        # 3) срок: проверяем для продуктов с заданным диапазоном (>0)
        if term and product.get("term_max_months", 0) > 0:
            if not (product["term_min_months"] <= term <= product["term_max_months"]):
                return False

        return True

    # --- объяснение (Шаг 4) ---------------------------------------------------
    @staticmethod
    def _build_explanation(product, parsed, lang="ru"):
        from i18n import t
        parts = []
        cat = parsed.get("intent_category")
        amount = parsed.get("amount")
        term = parsed.get("term_months")

        if cat and cat != "не_определено" and product["category"] == cat:
            purpose = product.get("purpose_kz") if lang == "kz" else product.get("purpose")
            purpose = purpose or product.get("purpose", "")
            parts.append(f'{t("expl_goal", lang)} «{purpose}»')

        if amount and product.get("max_amount", 0) > 0:
            lo = format_kzt(product["min_amount"])
            hi = format_kzt(product["max_amount"])
            am = format_kzt(amount)
            if product["min_amount"] <= amount <= product["max_amount"]:
                parts.append(t("expl_amount_ok", lang, amount=am, lo=lo, hi=hi))
            else:
                parts.append(t("expl_amount_no", lang, amount=am, lo=lo, hi=hi))

        if term and product.get("term_max_months", 0) > 0:
            tmin = product["term_min_months"]
            tmax = product["term_max_months"]
            if tmin <= term <= tmax:
                parts.append(t("expl_term_ok", lang, term=term, tmin=tmin, tmax=tmax))
            else:
                parts.append(t("expl_term_no", lang, term=term, tmin=tmin, tmax=tmax))

        if not parts:
            return t("expl_fallback", lang)

        return t("expl_prefix", lang) + ", ".join(parts) + "."

    # --- расчёт платежа для карточки (Шаг 4.5) --------------------------------
    @staticmethod
    def _attach_payment(product, parsed):
        # калькулятор только для кредитных продуктов со ставкой > 0
        credit_cats = {"ипотека", "автокредит", "потреб_кредит"}
        if product["category"] not in credit_cats:
            return None
        rate = product.get("rate_numeric")
        if not rate:
            return None
        amount = parsed.get("amount") or product.get("min_amount")
        term = parsed.get("term_months") or product.get("term_min_months")
        # если запрошенный срок выходит за диапазон продукта — считаем платёж
        # по ближайшему допустимому сроку, иначе цифра вводит в заблуждение
        tmax = product.get("term_max_months", 0)
        tmin = product.get("term_min_months", 0)
        if tmax and term > tmax:
            term = tmax
        elif tmin and term < tmin:
            term = tmin
        return annuity_payment(amount, rate, term)

    # --- полностью ли продукт подходит по сумме И сроку -----------------------
    @staticmethod
    def _is_full_match(product, parsed):
        """True, если продукт той же категории, что запрос, и сумма/срок
        попадают в его лимиты. Продукт чужой категории не считается полным
        совпадением, даже если формально не противоречит сумме/сроку."""
        cat = parsed.get("intent_category")
        if cat and cat != "не_определено" and product["category"] != cat:
            return False
        amount = parsed.get("amount")
        term = parsed.get("term_months")
        if amount and product.get("max_amount", 0) > 0:
            if not (product["min_amount"] <= amount <= product["max_amount"]):
                return False
        if term and product.get("term_max_months", 0) > 0:
            if not (product["term_min_months"] <= term <= product["term_max_months"]):
                return False
        return True

    def suggest_alternative_category(self, parsed):
        """Если по запросу ничего не подошло точно — ищем категорию, где
        сумма и срок реально укладываются. Возвращает (категория, пример) или None."""
        amount = parsed.get("amount")
        term = parsed.get("term_months")
        seen = {}
        for p in self.catalog:
            ok_amount = (not amount or p.get("max_amount", 0) == 0
                         or p["min_amount"] <= amount <= p["max_amount"])
            ok_term = (not term or p.get("term_max_months", 0) == 0
                       or p["term_min_months"] <= term <= p["term_max_months"])
            if ok_amount and ok_term and p["category"] != parsed.get("intent_category"):
                seen.setdefault(p["category"], p)
        if not seen:
            return None
        # берём первую найденную категорию и пример продукта
        cat = next(iter(seen))
        return cat, seen[cat]

    def is_off_topic(self, parsed, threshold=0.35):
        """True, если запрос не относится к финансовой теме.

        Два сигнала вместе (оба должны сработать):
          1) правила не определили категорию кредита;
          2) в запросе нет финансовых слов-маркеров И dense-близость ниже порога.
        Второй сигнал детерминирован (слова) и не зависит только от модели —
        это надёжнее, чем один порог близости, который меняется от модели к модели.
        """
        cat = parsed.get("intent_category")
        if cat and cat != "не_определено":
            return False
        query = (parsed.get("goal_text") or parsed.get("raw_query", "")).lower()
        if not query:
            return True

        # явные НЕфинансовые товары/темы — сразу off-topic, даже если рядом есть «купить»
        off_topic_words = [
            "хлеб", "самокат", "велосипед", "телефон", "кот", "кошк", "собак", "еда",
            "продукт", "борщ", "пицц", "кофе", "цвет", "игрушк", "автомат", "билет",
            "погод", "новост", "анекдот", "стих", "песн", "рецепт",
        ]
        if any(w in query for w in off_topic_words):
            return True

        # финансовые слова-маркеры: если хоть один есть — запрос по теме
        finance_markers = [
            "кредит", "займ", "заём", "деньг", "ипотек", "рефинанс", "автокредит",
            "наличн", "рассрочк", "ссуд", "взять", "одолж", "миллион", "млн", "тыс",
            "тенге", "₸", "процент", "ставк", "платёж", "платеж", "несие", "ақша",
            "қарыз", "квартир", "жиль", "машин", "авто", "образован", "лечени", "ремонт",
        ]
        has_finance_word = any(m in query for m in finance_markers)
        if has_finance_word:
            return False

        # нет финансовых слов и категория не ясна — проверяем семантику (raw dense)
        sims = self._dense_scores(query)
        if len(sims) == 0:
            return True
        return float(max(sims)) < threshold

    def popular_in_category(self, category, top_k=3):
        """Популярные продукты категории — когда параметров мало для точного подбора.
        Прокси популярности: для кредитов/ипотеки — ниже ставка лучше;
        для остального — порядок по каталогу (первые = более известные банки)."""
        items = [p for p in self.catalog if p["category"] == category]
        if not items:
            return []
        rate_cats = {"ипотека", "автокредит", "потреб_кредит", "кредитная_карта"}
        if category in rate_cats:
            items = sorted(items, key=lambda p: p.get("rate_numeric", 999) or 999)
        return items[:top_k]

    def similar_products(self, product_id, top_k=3):
        """Похожие продукты по близости эмбеддингов (item-based).

        Не требует истории пользователей: сходство считается между самими
        продуктами. Показывается как «вместе с этим часто смотрят».
        """
        # индекс исходного продукта
        idx = None
        for i, p in enumerate(self.catalog):
            if p["id"] == product_id:
                idx = i
                break
        if idx is None or self._matrix is None:
            return []

        # близость исходного продукта ко всем остальным
        import numpy as np
        if self.backend == "st":
            base = self._matrix[idx]
            sims = (self._matrix @ base.T).ravel()
        else:
            # TF-IDF матрица разреженная
            base = self._matrix[idx]
            sims = (self._matrix @ base.T).toarray().ravel()

        order = np.argsort(sims)[::-1]
        result = []
        for j in order:
            if j == idx:
                continue
            result.append(self.catalog[j])
            if len(result) >= top_k:
                break
        return result

    def filter_by_income(self, products, income):
        """Оставляет продукты, формальное требование к доходу которых не выше
        указанного. ЭТО НЕ СКОРИНГ: оценка кредитоспособности остаётся за банком,
        здесь только фильтр по формальному требованию min_income."""
        if not income:
            return products
        ok = []
        for p in products:
            req = p.get("requirements", {})
            min_income = req.get("min_income", 0) if isinstance(req, dict) else 0
            if not min_income or income >= min_income:
                ok.append(p)
        return ok

    # --- основной метод -------------------------------------------------------
    def recommend(self, parsed, top_k=3):
        """parsed — результат extract.extract(). Возвращает список карточек."""
        query = parsed.get("goal_text") or parsed.get("raw_query", "")
        from i18n import detect_lang
        lang = detect_lang(query)

        # гибрид: если правила не уверены в категории — уточняем эмбеддингами
        if (self._intent_clf is not None and self._intent_clf.available()
                and not self.is_off_topic(parsed)):
            from intent_classifier import combine
            emb_cat, emb_conf = self._intent_clf.classify(query)
            new_cat, new_conf, source = combine(
                parsed.get("intent_category"), parsed.get("intent_confidence", 0),
                emb_cat, emb_conf)
            if source == "embeddings":
                parsed = dict(parsed)
                parsed["intent_category"] = new_cat
                parsed["intent_confidence"] = max(0.6, new_conf)  # порог для фильтра
                parsed["intent_source"] = "embeddings"

        sims = self._similarities(query)

        # применяем фильтр
        candidates = []
        for i, product in enumerate(self.catalog):
            if self._passes_filter(product, parsed):
                candidates.append((i, float(sims[i])))

        # если фильтр отсёк всё — смягчаем: убираем фильтр по сумме/сроку,
        # оставляем только семантику (чтобы система не молчала)
        if not candidates:
            candidates = [(i, float(sims[i])) for i in range(len(self.catalog))]

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        results = []
        for i, score in top:
            product = self.catalog[i]
            results.append({
                "product": product,
                "score": round(score, 3),
                "explanation": self._build_explanation(product, parsed, lang),
                "payment": self._attach_payment(product, parsed),
                "full_match": self._is_full_match(product, parsed),
            })
        return results


if __name__ == "__main__":
    from extract import extract

    rec = Recommender()
    print(f"Бэкенд векторизации: {rec.backend}\n")

    q = "хочу машину за 5 млн на 3 года"
    parsed = extract(q)
    print("Запрос:", q)
    for r in rec.recommend(parsed):
        p = r["product"]
        print(f"  [{r['score']}] {p['bank']} — {p['name']}")
        print(f"      {r['explanation']}")
        if r["payment"]:
            print(f"      Платёж: {format_kzt(r['payment']['monthly_payment'])}/мес, "
                  f"переплата {format_kzt(r['payment']['overpayment'])}")
