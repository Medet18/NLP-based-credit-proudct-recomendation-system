# -*- coding: utf-8 -*-
"""
calculator.py — расчёт аннуитетного ежемесячного платежа по кредиту.

Формула: платёж = P * r * (1+r)^n / ((1+r)^n - 1)
  P — сумма кредита, r — месячная ставка (годовая/12/100), n — срок в месяцах.
"""


def annuity_payment(principal, annual_rate_percent, term_months):
    """Возвращает dict с расчётом или None, если данных не хватает.

    principal           — сумма кредита, тенге
    annual_rate_percent — годовая ставка в процентах (например 21.0)
    term_months         — срок в месяцах
    """
    if not principal or not term_months or annual_rate_percent is None:
        return None
    if principal <= 0 or term_months <= 0:
        return None

    r = annual_rate_percent / 12 / 100  # месячная ставка в долях

    if r == 0:  # беспроцентный случай (например рассрочка 0%)
        monthly = principal / term_months
    else:
        factor = (1 + r) ** term_months
        monthly = principal * r * factor / (factor - 1)

    total = monthly * term_months
    overpay = total - principal

    return {
        "monthly_payment": round(monthly),
        "total_payment": round(total),
        "overpayment": round(overpay),
        "principal": principal,
        "annual_rate": annual_rate_percent,
        "term_months": term_months,
    }


def format_kzt(amount):
    """Формат суммы в тенге с разделением разрядов: 123456 -> '123 456 ₸'."""
    return f"{amount:,.0f}".replace(",", " ") + " ₸"


if __name__ == "__main__":
    # хочу кредит 3 млн на 24 месяца под 24%
    res = annuity_payment(3_000_000, 24.0, 24)
    print("Кредит 3 млн, 24 мес, 24% годовых:")
    print("  Ежемесячный платёж:", format_kzt(res["monthly_payment"]))
    print("  Всего выплат:      ", format_kzt(res["total_payment"]))
    print("  Переплата:         ", format_kzt(res["overpayment"]))
