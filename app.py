# -*- coding: utf-8 -*-
"""
app.py — интерфейс «Персональный финансовый помощник на основе NLP».

Refined fintech дизайн. Вкладки: Помощник, Калькулятор, Админ, Аналитика, Результаты.
Каталог и фидбэк — в базе SQLite (db.py). Логика подбора без изменений.
"""

import os
import json
import html
import csv
import streamlit as st

from extract import extract, missing_fields_for_credit, parse_amount, parse_term_months
from recommender import Recommender, ST_MODEL_NAME
from calculator import format_kzt, annuity_payment
import db

st.set_page_config(page_title="Финансовый помощник", page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

# гарантируем, что база и таблицы существуют
db.init_db()


def load_css():
    path = os.path.join("assets", "style.css")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


@st.cache_resource
def load_recommender():
    return Recommender()

rec = load_recommender()

CAT_ICON = {
    "ипотека": "🏠", "автокредит": "🚗", "рефинансирование": "🔄",
    "микрокредит": "⚡", "целевой": "🎓", "потреб_кредит": "💵",
}
CATEGORIES = ["потреб_кредит", "ипотека", "автокредит", "рефинансирование",
              "микрокредит", "целевой"]

EXAMPLES_RU = [
    "хочу машину за 5 млн на 3 года",
    "нужны деньги на ремонт квартиры, 2 миллиона",
    "ипотека на квартиру по госпрограмме",
    "хочу объединить несколько кредитов в один",
    "нужен кредит на образование",
]
EXAMPLES_KZ = [
    "3 жылға 5 миллионға машина алғым келеді",
    "пәтерді жөндеуге 2 миллион ақша керек",
    "мемлекеттік бағдарлама бойынша пәтерге ипотека",
    "бірнеше несиені біріктіргім келеді",
    "оқуға несие керек",
]
EXAMPLES = EXAMPLES_RU  # значение по умолчанию (совместимость)


# ============================================================
#  ВКЛАДКА: ПОМОЩНИК
# ============================================================
def render_hero(lang="ru"):
    from i18n import t
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-mark"><span class="hero-dot"></span> FinAssist · NLP</div>
          <div class="hero-title">{t("hero_title", lang)}</div>
          <div class="hero-sub">{t("hero_sub", lang)}</div>
        </div>
        """, unsafe_allow_html=True)


def card_html(rank, r, lang="ru"):
    from i18n import t
    p = r["product"]
    icon = CAT_ICON.get(p["category"], "•")
    name_val = p.get("name_kz", p["name"]) if lang == "kz" else p["name"]
    bank = html.escape(p["bank"]); name = html.escape(name_val)
    conds = []
    if p.get("max_amount", 0) > 0:
        conds.append(f'<div class="cond-row"><b>{t("amount", lang)}:</b> '
                     f'<span class="mono">{format_kzt(p["min_amount"])} – {format_kzt(p["max_amount"])}</span></div>')
    if p.get("term_max_months", 0) > 0:
        conds.append(f'<div class="cond-row"><b>{t("term", lang)}:</b> '
                     f'<span class="mono">{p["term_min_months"]}–{p["term_max_months"]} {t("months", lang)}</span></div>')
    conds.append(f'<div class="cond-row"><b>{t("rate", lang)}:</b> {html.escape(str(p.get("rate_or_yield","")))}</div>')
    conds_html = "".join(conds)
    pct = max(4, min(100, int(round(r["score"] * 100))))
    expl = html.escape(r["explanation"])
    expl_html = (f'<div class="explain"><div class="crit"><span class="check">✓</span>'
                 f'<span>{expl}</span></div></div>')
    calc_html = ""
    if r["payment"]:
        pay = r["payment"]
        calc_html = (f'<div class="calc-line">{t("payment", lang)} ≈ <span class="big">{format_kzt(pay["monthly_payment"])}</span>/{t("per_month", lang)} '
                     f'· {t("overpay", lang)} {format_kzt(pay["overpayment"])}</div>')
    return f"""
    <div class="card" style="animation-delay:{(rank-1)*0.08}s">
      <div class="card-top"><span class="bank-badge">{bank}</span><span class="cat-icon">{icon}</span></div>
      <div class="card-title">{name}</div>
      {conds_html}
      <div class="rel-wrap"><div class="rel-label">{t("relevance", lang)} · {r['score']}</div>
        <div class="rel-bar"><div class="rel-fill" style="width:{pct}%"></div></div></div>
      {expl_html}{calc_html}
    </div>"""


@st.dialog("Подробнее о продукте")
def show_product_dialog(p, lang="ru"):
    from i18n import t
    name_val = p.get("name_kz", p["name"]) if lang == "kz" else p["name"]
    st.markdown(f"### {p['bank']} — {name_val}")
    cat_label = t(f"cat_{p['category']}", lang)
    st.caption(f"{t('goal', lang)}: {cat_label}")
    # описание: казахское назначение при kz, иначе описание
    desc = p.get("purpose_kz", "") if lang == "kz" else p.get("description_text", "")
    st.write(desc or p.get("description_text", ""))

    rows = []
    if p.get("max_amount", 0) > 0:
        rows.append((t("amount", lang), f'{format_kzt(p["min_amount"])} – {format_kzt(p["max_amount"])}'))
    if p.get("term_max_months", 0) > 0:
        rows.append((t("term", lang), f'{p["term_min_months"]}–{p["term_max_months"]} {t("months", lang)}'))
    rows.append((t("rate", lang), p.get("rate_or_yield", "—")))
    if p.get("benefits"):
        rows.append(("Выгоды" if lang == "ru" else "Артықшылықтары", p["benefits"]))
    if p.get("fees"):
        rows.append(("Комиссии" if lang == "ru" else "Комиссиялар", p["fees"]))
    rows.append(("Залог" if lang == "ru" else "Кепіл",
                 ("требуется" if lang == "ru" else "қажет") if p.get("collateral_required")
                 else ("не требуется" if lang == "ru" else "қажет емес")))
    for label, val in rows:
        st.markdown(f"**{label}:** {val}")

    url = p.get("bank_url")
    if url:
        link_text = "Перейти на сайт банка →" if lang == "ru" else "Банк сайтына өту →"
        st.markdown(f'<a href="{url}" target="_blank" '
                    f'style="display:inline-block;margin-top:12px;padding:10px 18px;'
                    f'background:#115E4A;color:#fff;border-radius:12px;text-decoration:none;'
                    f'font-weight:600;">{link_text}</a>', unsafe_allow_html=True)

    # похожие продукты (item-based)
    similar = rec.similar_products(p["id"], top_k=3)
    if similar:
        st.markdown("---")
        st.markdown("**Похожие продукты:**" if lang == "ru" else "**Ұқсас өнімдер:**")
        for sp in similar:
            sp_name = sp.get("name_kz", sp["name"]) if lang == "kz" else sp["name"]
            st.markdown(f"• {sp['bank']} — {sp_name}")


def render_results(parsed, query, ui_lang=None):
    from i18n import detect_lang, t
    # приоритет — выбранный язык страницы; если не задан, определяем по запросу
    lang = ui_lang or detect_lang(query)
    results = rec.recommend(parsed, top_k=10)   # берём до 10, показываем частями
    db.log_query(query, {
        "intent": parsed["intent_category"], "amount": parsed["amount"],
        "term": parsed["term_months"]}, found=bool(results))
    if not results:
        st.markdown('<div class="soft-note">Не удалось подобрать продукты под этот запрос. '
                    'Попробуйте переформулировать или уточнить цель.</div>', unsafe_allow_html=True)
        return

    # Вариант А: категория ясна, но сумма и срок не заданы —
    # показываем результаты как популярные в категории + честная пометка
    cat0 = parsed.get("intent_category")
    if (cat0 and cat0 != "не_определено"
            and not parsed.get("amount") and not parsed.get("term_months")):
        st.markdown(
            f'<div class="soft-note">Вы не указали сумму и срок — показываю популярные '
            f'продукты категории <b>«{html.escape(cat0)}»</b>. Для точного подбора и расчёта '
            f'платежа укажите сумму и срок.</div>', unsafe_allow_html=True)

    # есть ли хоть один продукт, точно подходящий по сумме И сроку?
    has_full = any(r.get("full_match") for r in results[:3])
    if not has_full and (parsed.get("amount") or parsed.get("term_months")):
        cat = parsed.get("intent_category", "")
        amount_s = format_kzt(parsed["amount"]) if parsed["amount"] else "—"
        term_s = f'{parsed["term_months"]} мес' if parsed["term_months"] else "—"
        msg = (f'По запросу <b>{amount_s}</b> на <b>{term_s}</b> среди продуктов категории '
               f'«{html.escape(cat)}» нет точного совпадения по сумме и сроку.')
        alt = rec.suggest_alternative_category(parsed)
        if alt:
            alt_cat, alt_prod = alt
            msg += (f' Для таких параметров лучше подойдёт категория <b>«{html.escape(alt_cat)}»</b> '
                    f'— например, «{html.escape(alt_prod["bank"])} · {html.escape(alt_prod["name"])}». '
                    f'Уточните запрос под эту цель или измените сумму/срок.')
        else:
            msg += ' Попробуйте изменить сумму или срок. Ниже — ближайшие по смыслу варианты.'
        st.markdown(f'<div class="soft-note">{msg}</div>', unsafe_allow_html=True)

    # сколько карточек показываем сейчас (хранится между перерисовками)
    if "show_count" not in st.session_state:
        st.session_state["show_count"] = 3
    shown = min(st.session_state["show_count"], len(results))

    st.markdown(f'<div class="section-h">{t("recommendations", lang)}</div>', unsafe_allow_html=True)

    # карточки рядами по 3; всегда 3 колонки, чтобы одиночная карточка
    # в неполном ряду не растягивалась на всю ширину
    visible = results[:shown]
    for row_start in range(0, len(visible), 3):
        row = visible[row_start:row_start + 3]
        cols = st.columns(3)
        for offset in range(3):
            with cols[offset]:
                if offset >= len(row):
                    continue          # пустая колонка — карточки держат ширину в треть
                r = row[offset]
                idx = row_start + offset + 1
                st.markdown(card_html(idx, r, lang), unsafe_allow_html=True)
                pid = r["product"]["id"]
                if st.button("Подробнее" if lang == "ru" else "Толығырақ",
                             key=f"more_{idx}_{pid}", width='stretch'):
                    show_product_dialog(r["product"], lang)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("👍", key=f"up_{idx}_{pid}", width='stretch'):
                        db.save_feedback(pid, query, +1); st.toast("Спасибо за оценку!")
                with b2:
                    if st.button("👎", key=f"down_{idx}_{pid}", width='stretch'):
                        db.save_feedback(pid, query, -1); st.toast("Спасибо за оценку!")

    # кнопка «показать ещё» — если есть что показывать сверх текущих
    if shown < len(results):
        remaining = len(results) - shown
        if st.button((f"Показать ещё ({remaining})" if lang == "ru" else f"Тағы көрсету ({remaining})"),
                     key="show_more", width='stretch'):
            st.session_state["show_count"] = min(shown + 3, 10, len(results))
            st.rerun()

    cat = parsed["intent_category"]
    cat_disp = t(f"cat_{cat}", lang) if cat and cat != "не_определено" else cat
    amount = format_kzt(parsed["amount"]) if parsed["amount"] else "—"
    term = f'{parsed["term_months"]} мес' if parsed["term_months"] else "—"
    st.markdown(f'<div class="section-h" style="font-size:18px; margin-top:24px;">{t("understood", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"""<div class="understood">
          <div class="u-chip">{t("goal", lang)}: <b>{html.escape(cat_disp)}</b></div>
          <div class="u-chip">{t("amount", lang)}: <b>{amount}</b></div>
          <div class="u-chip">{t("term", lang)}: <b>{term}</b></div>
          <div class="u-chip">{t("confidence", lang)}: <b>{parsed['intent_confidence']}</b></div>
        </div>""", unsafe_allow_html=True)

    # Explainable AI: визуальное сравнение показанных продуктов
    if len(visible) >= 2:
        with st.expander("Сравнить продукты — почему один лучше другого"):
            render_comparison(visible, parsed)


def render_comparison(items, parsed):
    """Таблица сравнения продуктов с подсветкой лучших значений по критериям.
    Это объяснимость (explainable recommendation): видно, по каким параметрам
    каждый продукт выигрывает."""
    rows = []
    for r in items:
        p = r["product"]
        rows.append({
            "Продукт": f'{p["bank"]} · {p["name"]}',
            "Релевантность": r["score"],
            "Ставка": p.get("rate_numeric", 0) or 0,
            "Макс. сумма": p.get("max_amount", 0),
            "Макс. срок (мес)": p.get("term_max_months", 0),
        })

    # определяем лучшие значения: ставка — меньше лучше, остальное — больше лучше
    best = {
        "Релевантность": max(r["Релевантность"] for r in rows),
        "Ставка": min((r["Ставка"] for r in rows if r["Ставка"] > 0), default=0),
        "Макс. сумма": max(r["Макс. сумма"] for r in rows),
        "Макс. срок (мес)": max(r["Макс. срок (мес)"] for r in rows),
    }

    # рисуем HTML-таблицу с подсветкой лучших ячеек
    cols = ["Продукт", "Релевантность", "Ставка", "Макс. сумма", "Макс. срок (мес)"]
    head = "".join(f"<th style='text-align:left;padding:8px 12px;border-bottom:2px solid #115E4A;'>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = ""
        for c in cols:
            val = r[c]
            disp = val
            if c == "Ставка" and val: disp = f"{val}%"
            elif c == "Макс. сумма" and val: disp = format_kzt(val)
            highlight = (c in best and val == best[c] and val)
            style = "padding:8px 12px;border-bottom:1px solid #E7E2D8;"
            if highlight:
                style += "background:#E6F2EC;font-weight:600;color:#115E4A;"
            cells += f"<td style='{style}'>{disp}</td>"
        body += f"<tr>{cells}</tr>"
    st.markdown(f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>",
                unsafe_allow_html=True)
    st.caption("Зелёным выделено лучшее значение по каждому критерию: выше релевантность, "
               "ниже ставка, больше лимит суммы и срока.")


def tab_assistant():
    from i18n import t
    # выбор языка страницы — влияет на всю вкладку
    ui_lang_label = st.radio("Язык / Тіл", ["Русский", "Қазақша"],
                             horizontal=True, key="ui_lang")
    lang = "kz" if ui_lang_label == "Қазақша" else "ru"

    render_hero(lang)
    backend_note = (("мультиязычные эмбеддинги" if lang == "ru" else "мультитілді эмбеддингтер")
                    if rec.backend == "st" else ("TF-IDF, офлайн" if lang == "ru" else "TF-IDF, офлайн"))
    st.markdown(f'<div class="hero-sub" style="margin-bottom:14px;">{t("engine", lang)}: '
                f'<b>{rec.backend}</b> ({backend_note})</div>', unsafe_allow_html=True)

    query = st.text_input("query", placeholder=t("query_ph", lang),
                          key="main_query", label_visibility="collapsed")
    go = st.button(t("search_btn", lang), key="go_main")
    if go:
        st.session_state["show_count"] = 3   # новый поиск — снова показываем 3

    examples = EXAMPLES_KZ if lang == "kz" else EXAMPLES_RU
    st.caption(t("search_examples_label", lang))
    chips = "".join(f'<span class="chip">{html.escape(e)}</span>' for e in examples)
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    st.caption(t("examples_hint", lang))

    if query.strip():
        from extract import is_help_query
        from i18n import t
        # lang уже определён выбором языка страницы выше

        # 1) вопрос о возможностях системы -> справка
        if is_help_query(query):
            st.info(f'**{t("help_title", lang)}**\n\n{t("help_body", lang)}')
            return

        parsed = extract(query)

        # 2) запрос не по финансовой теме -> предупреждение (красное), примеры уже выше
        if rec.is_off_topic(parsed):
            st.error(t("off_topic", lang))
            return

        missing = missing_fields_for_credit(parsed)
        if missing:
            st.markdown(f'<div class="soft-note">Чтобы подобрать точнее, уточните: '
                        f'<b>{", ".join(missing)}</b>. Заполните поля ниже или нажмите '
                        f'«Подобрать продукты» — система подберёт по смыслу запроса.</div>',
                        unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                ea = st.text_input("Сумма (тенге или «3 млн»)",
                                   value="" if parsed["amount"] is None else str(parsed["amount"]),
                                   key="extra_amount")
            with cc2:
                et = st.text_input("Срок (месяцев)",
                                   value="" if parsed["term_months"] is None else str(parsed["term_months"]),
                                   key="extra_term")
            if ea.strip():
                a = parse_amount(ea)
                if a: parsed["amount"] = a
            if et.strip():
                tm = parse_term_months(et + " месяцев") or parse_term_months(et)
                if tm: parsed["term_months"] = tm
        # показываем результаты, пока в поле есть запрос (не только в момент нажатия
        # кнопки), иначе нажатие 👍/Подробнее/Показать ещё перезапускает скрипт и
        # результаты исчезают
        render_results(parsed, query, lang)
    elif go:
        st.markdown('<div class="soft-note">Введите запрос, чтобы начать подбор.</div>',
                    unsafe_allow_html=True)


# ============================================================
#  ВКЛАДКА: КАЛЬКУЛЯТОР С ГРАФИКОМ АМОРТИЗАЦИИ (Блок 4)
# ============================================================
def amortization_schedule(principal, annual_rate, term_months):
    """Помесячный график аннуитета: тело, проценты, остаток, накопленная переплата."""
    A = annuity_payment(principal, annual_rate, term_months)["monthly_payment"]
    r = annual_rate / 12 / 100
    balance = principal
    cum_over = 0
    rows = []
    for m in range(1, term_months + 1):
        interest = balance * r
        principal_part = A - interest
        if m == term_months:           # последний платёж гасит остаток полностью
            principal_part = balance
        balance = max(0, balance - principal_part)
        cum_over += interest
        rows.append({
            "Месяц": m, "Платёж": round(A), "Тело": round(principal_part),
            "Проценты": round(interest), "Остаток долга": round(balance),
            "Накопленная переплата": round(cum_over),
        })
    return rows


def tab_calculator():
    st.markdown('<div class="section-h">Кредитный калькулятор</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.slider("Сумма, ₸", 100_000, 30_000_000, 3_000_000, step=100_000)
    with c2:
        term = st.slider("Срок, мес", 6, 240, 24, step=6)
    with c3:
        rate = st.slider("Ставка, % годовых", 5.0, 35.0, 21.0, step=0.5)

    res = annuity_payment(amount, rate, term)
    if not res:
        return
    k1, k2, k3 = st.columns(3)
    k1.metric("Платёж в месяц", format_kzt(res["monthly_payment"]))
    k2.metric("Всего выплат", format_kzt(res["total_payment"]))
    k3.metric("Переплата", format_kzt(res["overpayment"]))

    sched = amortization_schedule(amount, rate, term)

    st.markdown('<div class="section-h" style="font-size:18px;">Остаток долга по месяцам</div>',
                unsafe_allow_html=True)
    st.line_chart({"Остаток долга": [row["Остаток долга"] for row in sched]})

    st.markdown('<div class="section-h" style="font-size:18px;">Разбивка платежа: тело и проценты</div>',
                unsafe_allow_html=True)
    st.bar_chart({
        "Тело": [row["Тело"] for row in sched],
        "Проценты": [row["Проценты"] for row in sched],
    })

    st.markdown('<div class="section-h" style="font-size:18px;">Накопленная переплата</div>',
                unsafe_allow_html=True)
    st.line_chart({"Накопленная переплата": [row["Накопленная переплата"] for row in sched]})

    with st.expander("Таблица по месяцам"):
        st.dataframe(sched, width='stretch', hide_index=True)


# ============================================================
#  ВКЛАДКА: АДМИН (Блок 2) — создать/изменить/удалить
# ============================================================
def validate_product(d):
    errors = []
    if not d["id"].strip(): errors.append("ID обязателен")
    if not d["bank"].strip(): errors.append("Банк обязателен")
    if not d["name"].strip(): errors.append("Название обязательно")
    if d["min_amount"] < 0 or d["max_amount"] < 0: errors.append("Суммы не могут быть отрицательными")
    if d["max_amount"] and d["min_amount"] > d["max_amount"]:
        errors.append("Минимальная сумма больше максимальной")
    if d["term_min_months"] > d["term_max_months"] and d["term_max_months"] > 0:
        errors.append("Минимальный срок больше максимального")
    if not (0 <= d["rate_numeric"] <= 100): errors.append("Ставка должна быть от 0 до 100")
    if not d["description_text"].strip(): errors.append("Описание обязательно (по нему идёт поиск)")
    return errors


def recompute_embedding(product):
    """Пересчитать эмбеддинг продукта и записать в базу (если модель доступна)."""
    if rec.backend != "st" or rec._st_model is None:
        return False
    text = rec._product_text(product)
    vec = rec._st_model.encode([text], normalize_embeddings=True)[0]
    db.save_embedding(product["id"], vec, ST_MODEL_NAME)
    return True


def tab_admin():
    st.markdown('<div class="section-h">Управление каталогом</div>', unsafe_allow_html=True)
    products = db.get_all_products()
    st.caption(f"Продуктов: {len(products)} · банков: {len(set(p['bank'] for p in products))} "
               f"· категорий: {len(set(p['category'] for p in products))}")

    # фильтр + просмотр
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        cats = ["все"] + sorted(set(p["category"] for p in products))
        sel = st.selectbox("Категория", cats, key="admin_cat")
    with fcol2:
        search = st.text_input("Поиск по названию/банку", key="admin_search")

    rows = products
    if sel != "все":
        rows = [p for p in rows if p["category"] == sel]
    if search.strip():
        s = search.lower()
        rows = [p for p in rows if s in p["name"].lower() or s in p["bank"].lower()]

    st.dataframe(
        [{"ID": p["id"], "Банк": p["bank"], "Название": p["name"],
          "Категория": p["category"], "Ставка": p.get("rate_numeric"),
          "Обновлён": p.get("updated_at", "")} for p in rows],
        width='stretch', hide_index=True)

    st.divider()

    # форма создать/изменить
    st.markdown('<div class="section-h" style="font-size:18px;">Создать / изменить продукт</div>',
                unsafe_allow_html=True)
    ids = ["➕ новый продукт"] + [p["id"] for p in products]
    chosen = st.selectbox("Выберите продукт для редактирования или создайте новый", ids, key="admin_edit_id")

    editing = None if chosen == "➕ новый продукт" else db.get_product(chosen)

    def gv(field, default=""):
        return editing.get(field, default) if editing else default
    
    # при смене выбранного продукта (или после сохранения) заполняем поля
    # значениями продукта через session_state, затем виджеты читают их по ключу
    if st.session_state.get("_last_chosen") != chosen:
        st.session_state["f_id"] = gv("id")
        st.session_state["f_bank"] = gv("bank")
        st.session_state["f_name"] = gv("name")
        st.session_state["f_cat"] = gv("category", "ипотека") if gv("category") in CATEGORIES else "ипотека"
        st.session_state["f_min"] = int(gv("min_amount", 0) or 0)
        st.session_state["f_max"] = int(gv("max_amount", 0) or 0)
        st.session_state["f_tmin"] = int(gv("term_min_months", 0) or 0)
        st.session_state["f_tmax"] = int(gv("term_max_months", 0) or 0)
        st.session_state["f_rate"] = float(gv("rate_numeric", 0) or 0)
        st.session_state["f_rate_str"] = gv("rate_or_yield")
        st.session_state["f_purpose"] = gv("purpose")
        st.session_state["f_desc"] = gv("description_text")
        st.session_state["f_benefits"] = gv("benefits")
        st.session_state["f_url"] = gv("bank_url") or gv("source_url")
        st.session_state["_last_chosen"] = chosen

    e1, e2, e3 = st.columns(3)
    with e1:
        f_id = st.text_input("ID", value=gv("id"), disabled=bool(editing), key="f_id")
        f_bank = st.text_input("Банк", value=gv("bank"), key="f_bank")
        f_name = st.text_input("Название", value=gv("name"), key="f_name")
        f_cat = st.selectbox("Категория", CATEGORIES,
                             index=CATEGORIES.index(gv("category", "ипотека")) if gv("category") in CATEGORIES else 0,
                             key="f_cat")
    with e2:
        f_min = st.number_input("Мин. сумма", min_value=0, value=int(gv("min_amount", 0)), step=100000, key="f_min")
        f_max = st.number_input("Макс. сумма", min_value=0, value=int(gv("max_amount", 0)), step=100000, key="f_max")
        f_tmin = st.number_input("Срок мин (мес)", min_value=0, value=int(gv("term_min_months", 0)), key="f_tmin")
        f_tmax = st.number_input("Срок макс (мес)", min_value=0, value=int(gv("term_max_months", 0)), key="f_tmax")
    with e3:
        f_rate = st.number_input("Ставка, % (первоначальная)", min_value=0.0, max_value=100.0,
                                 step=0.5, key="f_rate")
        f_rate_str = st.text_input("ГЭСВ (текст, как на сайте банка)", key="f_rate_str")
        f_purpose = st.text_input("Назначение", value=gv("purpose"), key="f_purpose")
        f_url = st.text_input("Ссылка на офиц. сайт продукта", key="f_url",
                              placeholder="например: https://kaspi.kz/credit")
    f_desc = st.text_area("Описание (по нему идёт семантический поиск)", value=gv("description_text"), key="f_desc")
    f_benefits = st.text_input("Выгоды", value=gv("benefits"), key="f_benefits")

    save_col, del_col = st.columns([1, 1])
    with save_col:
        if st.button("💾 Сохранить продукт", key="admin_save", width='stretch'):
            # при редактировании ID берём из выбранного продукта (поле disabled),
            # при создании — из введённого значения
            product_id = editing["id"] if editing else f_id
            # нормализуем ссылку: дополняем https://, если не указан протокол
            url_val = f_url.strip()
            if url_val and not url_val.startswith("http"):
                url_val = "https://" + url_val
            # для текстовых полей — fallback на значение продукта, если поле пустое
            def fv(widget_val, field):
                return widget_val if str(widget_val).strip() else gv(field)
            product = {
                "id": product_id,
                "bank": fv(f_bank, "bank"), "name": fv(f_name, "name"), "category": f_cat,
                "purpose": fv(f_purpose, "purpose"), "currency": "KZT",
                "min_amount": int(f_min), "max_amount": int(f_max),
                "term_min_months": int(f_tmin), "term_max_months": int(f_tmax),
                "rate_numeric": float(f_rate), "rate_or_yield": f_rate_str,
                "collateral_required": gv("collateral_required", False),
                "description_text": fv(f_desc, "description_text"), "benefits": fv(f_benefits, "benefits"),
                "fees": gv("fees", ""),
                "requirements": gv("requirements", {}), "attributes": gv("attributes", {}),
                "source_url": gv("source_url", "manual"),
                "bank_url": url_val,
                "data_source": gv("data_source", "manual"),
            }
            errors = validate_product(product)
            if errors:
                st.error("Не сохранено:\n- " + "\n- ".join(errors))
            else:
                db.upsert_product(product)
                emb_ok = recompute_embedding(product)
                load_recommender.clear()  # сбросить кэш, чтобы поиск увидел изменения
                msg = "Продукт сохранён в базу."
                msg += " Эмбеддинг пересчитан." if emb_ok else " Эмбеддинг будет пересчитан при наличии модели."
                # очищаем поля формы, чтобы подгрузились актуальные значения из базы
                for k in ["f_id","f_bank","f_name","f_cat","f_min","f_max","f_tmin",
                          "f_tmax","f_rate","f_rate_str","f_purpose","f_desc","f_benefits",
                           "f_url","_last_chosen","admin_edit_id"]:
                    st.session_state.pop(k, None)
                st.toast(msg); st.rerun()
    with del_col:
        if editing and st.button("🗑 Удалить продукт", key="admin_del", width='stretch'):
            db.delete_product(editing["id"])
            load_recommender.clear()
            st.toast("Продукт удалён"); st.rerun()

    st.divider()
    if st.button("⬇ Экспорт каталога в JSON", key="admin_export"):
        data = db.get_all_products()
        path = "catalog_export.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        st.success(f"Каталог выгружен в {path} ({len(data)} продуктов)")


# ============================================================
#  ВКЛАДКА: АНАЛИТИКА (Блок 3)
# ============================================================
def tab_analytics():
    st.markdown('<div class="section-h">Аналитика использования</div>', unsafe_allow_html=True)
    qs = db.get_query_stats()
    a1, a2, a3 = st.columns(3)
    a1.metric("Всего запросов", qs["total"])
    a2.metric("Не нашлось решений", qs["not_found"])
    a3.metric("Доля «не нашлось»", f'{qs["not_found_rate"]*100:.0f}%')

    st.markdown('<div class="section-h" style="font-size:18px;">Последние запросы</div>',
                unsafe_allow_html=True)
    recent = db.get_recent_queries(15)
    if recent:
        st.dataframe(
            [{"Запрос": r["query"], "Нашлось": "да" if r["found"] else "нет",
              "Время": r["created_at"]} for r in recent],
            width='stretch', hide_index=True)
    else:
        st.caption("Пока нет запросов. Сделайте поиск во вкладке «Помощник».")

    st.markdown('<div class="section-h" style="font-size:18px;">Оценки по продуктам 👍/👎</div>',
                unsafe_allow_html=True)
    fb = db.get_feedback_stats()
    if fb:
        st.dataframe(
            [{"Продукт": r["product_id"], "👍": r["up"], "👎": r["down"]} for r in fb],
            width='stretch', hide_index=True)
    else:
        st.caption("Пока нет оценок. Поставьте 👍/👎 на карточках рекомендаций.")


# ============================================================
#  ВКЛАДКА: РЕЗУЛЬТАТЫ
# ============================================================
def tab_results():
    st.markdown('<div class="section-h">Результаты экспериментов</div>', unsafe_allow_html=True)
    rd = "results"
    if not os.path.isdir(rd):
        st.markdown('<div class="soft-note">Папка results/ не найдена. Запустите '
                    '<b>python eval.py</b>.</div>', unsafe_allow_html=True)
        return
    for title, fname in [("Сравнение методов поиска", "comparison.csv"),
                         ("Вклад компонентов (ablation)", "ablation.csv"),
                         ("Мультиязычность (code-switching)", "multilang_comparison.csv")]:
        path = os.path.join(rd, fname)
        if os.path.exists(path):
            st.markdown(f'<div class="section-h" style="font-size:18px;">{title}</div>',
                        unsafe_allow_html=True)
            with open(path, encoding="utf-8") as f:
                st.dataframe(list(csv.DictReader(f)), width='stretch', hide_index=True)
    for img in ["comparison.png", "multilang_comparison.png"]:
        p = os.path.join(rd, img)
        if os.path.exists(p):
            st.image(p, width='stretch')


# ============================================================
#  РАСКЛАДКА: клиентская часть + админка за паролем в сайдбаре
# ============================================================

# Пароль админа (для прототипа — простой; в реальной системе хранился бы безопасно)
ADMIN_PASSWORD = "admin"

# статус админа хранится в session_state, чтобы переживать перерисовки
is_admin = st.session_state.get("is_admin", False)

if is_admin:
    # ОТДЕЛЬНЫЙ АДМИН-ЭКРАН на всю страницу (клиентская часть скрыта)
    top = st.columns([3, 1])
    with top[0]:
        st.markdown('<div class="hero-mark" style="margin-top:8px;">'
                    '<span class="hero-dot"></span> АДМИН-ПАНЕЛЬ</div>', unsafe_allow_html=True)
    with top[1]:
        if st.button("← Вернуться к помощнику", key="admin_back", width='stretch'):
            st.session_state["is_admin"] = False
            st.rerun()
    st.divider()
    at1, at2, at3 = st.tabs(["Управление продуктами", "Аналитика", "Результаты экспериментов"])
    with at1: tab_admin()
    with at2: tab_analytics()
    with at3: tab_results()
else:
    # КЛИЕНТСКАЯ ЧАСТЬ: только помощник и калькулятор
    t1, t2 = st.tabs(["Помощник", "Калькулятор"])
    with t1: tab_assistant()
    with t2: tab_calculator()

    # неприметный вход для администратора в самом низу
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    with st.expander("⚙ Служебный доступ", expanded=False):
        pw_col, _ = st.columns([1, 3])   # поле пароля — узкая колонка слева
        with pw_col:
            admin_pass = st.text_input("Пароль", type="password", key="admin_pass_bottom",
                                       label_visibility="collapsed", placeholder="Пароль администратора")
            if st.button("Войти", key="admin_login", width='stretch'):
                if admin_pass == ADMIN_PASSWORD:
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Неверный пароль")
