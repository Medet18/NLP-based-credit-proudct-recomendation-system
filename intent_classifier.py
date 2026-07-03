# -*- coding: utf-8 -*-
"""
intent_classifier.py — определение категории намерения через эмбеддинги.

Дополняет rule-based определение (extract.py): когда правила не уверены,
категория определяется семантически — по близости запроса к эталонным
описаниям каждой категории. Использует ту же модель MiniLM.

Гибрид: правила дают быстрый и точный результат на явных словах-маркерах,
эмбеддинги подхватывают случаи, где слов-маркеров нет, но смысл понятен.
"""

import numpy as np

# эталонные описания категорий (по ним строится «центр» каждой категории)
CATEGORY_ANCHORS = {
    "ипотека": "ипотека на покупку квартиры или дома, жильё, недвижимость, госпрограмма ипотеки",
    "автокредит": "кредит на покупку автомобиля, машина, авто новое или с пробегом",
    "рефинансирование": "рефинансирование кредитов, объединить кредиты, снизить ставку, перекредитоваться",
    "микрокредит": "небольшой займ до зарплаты, маленькая сумма быстро на короткий срок",
    "целевой": "кредит на образование или лечение, оплата обучения, целевой кредит",
    "потреб_кредит": "потребительский кредит наличными на любые цели, деньги на ремонт, под залог недвижимости",
}


class IntentClassifier:
    def __init__(self, st_model=None):
        self.model = st_model
        self.categories = list(CATEGORY_ANCHORS.keys())
        self._anchor_matrix = None
        if self.model is not None:
            texts = [CATEGORY_ANCHORS[c] for c in self.categories]
            self._anchor_matrix = np.asarray(
                self.model.encode(texts, normalize_embeddings=True))

    def available(self):
        return self._anchor_matrix is not None

    def classify(self, query):
        """Возвращает (категория, уверенность) по близости к эталонам.
        Если модель недоступна — (None, 0)."""
        if not self.available():
            return None, 0.0
        qv = np.asarray(self.model.encode([query], normalize_embeddings=True))
        sims = (self._anchor_matrix @ qv.T).ravel()
        best = int(np.argmax(sims))
        # уверенность — нормированная близость (косинус уже в [-1,1], берём как есть)
        confidence = round(float(sims[best]), 3)
        return self.categories[best], confidence


def combine(rule_category, rule_conf, emb_category, emb_conf, emb_threshold=0.35):
    """Объединяет результат правил и эмбеддингов.

    Логика гибрида:
      - если правила уверены (conf >= 0.6) — доверяем правилам;
      - иначе, если эмбеддинги дали уверенный ответ (>= threshold) — берём их;
      - иначе оставляем результат правил (может быть 'не_определено').
    """
    if rule_category and rule_category != "не_определено" and rule_conf >= 0.6:
        return rule_category, rule_conf, "rules"
    if emb_category and emb_conf >= emb_threshold:
        return emb_category, emb_conf, "embeddings"
    return rule_category, rule_conf, "rules"
