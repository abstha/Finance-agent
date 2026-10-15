import re
import warnings
from datetime import datetime

import pandas as pd
import pdfplumber

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

TEXT_DATE_RE = re.compile(
    r"^\s*(?P<day>\d{1,2})\s*"
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])"
    r"(?:\s*(?P<year>20\d{2})(?!\d))?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

SLASH_DATE_RE = re.compile(
    r"^\s*(?P<day>\d{1,2})/(?P<mon>\d{1,2})/(?P<year>\d{2,4})\s+(?P<rest>.*)$"
)

MONEY_RE = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?(?:\s?(?:CR|DR)\b)?", re.IGNORECASE)

STOP_HINTS = ("closingbalance", "balancebroughtforward", "balancecarriedforward")

X_TOLERANCES = (1, 1.5, 2, 3)
PAGE_RE = re.compile(r"\bpage\s*\d+\s*of\s*\d+\b", re.IGNORECASE)


def _to_float(token: str) -> float:
    """'$10.63 DR' -> -10.63 ; '(45.00)' -> -45.0 ; '$60.49 CR' -> 60.49"""
    t = token.strip()
    negative = t.startswith("(") or t.startswith("-") or t.upper().endswith("DR")
    return -float(re.sub(r"[^\d.]", "", t)) if negative else float(re.sub(r"[^\d.]", "", t))


def _match_date(line: str):
    """Return (day, month, explicit_year_or_None, rest_of_line) or None."""
    m = TEXT_DATE_RE.match(line)
    if m:
        year = int(m.group("year")) if m.group("year") else None
        return int(m.group("day")), MONTHS[m.group("mon")[:3].lower()], year, m.group("rest")
    m = SLASH_DATE_RE.match(line)
    if m:
        year = int(m.group("year"))
        year = year + 2000 if year < 100 else year
        return int(m.group("day")), int(m.group("mon")), year, m.group("rest")
    return None


def _parse_lines(lines):
    """Turn extracted text lines into a DataFrame, or None if no transactions are found."""
    all_text = "\n".join(lines)
    years = re.findall(r"\b(20\d{2})\b", all_text)
    fallback_year = int(max(set(years), key=years.count)) if years else datetime.now().year

    cur_year = None        
    last_month = None      
    opening_balance = None
    closing_balances = []  # used for a final integrity check
    records = []
    current = None         

    for line in lines:
        squash = line.lower().replace(" ", "")

        if "openingbalance" in squash:
            money = MONEY_RE.findall(line)
            if money and opening_balance is None:
                opening_balance = _to_float(money[-1])
            d = _match_date(line)
            if d and d[2] and cur_year is None:
                cur_year = d[2]
            current = None
            continue

        if any(h in squash for h in STOP_HINTS) or PAGE_RE.search(line):
            if "closingbalance" in squash:
                money = MONEY_RE.findall(line)
                if money:
                    closing_balances.append(_to_float(money[-1]))
            current = None
            continue

        d = _match_date(line)
        if d:
            day, month, explicit_year, text = d
            if cur_year is None:
                cur_year = explicit_year or fallback_year
            if explicit_year:
                cur_year = explicit_year
            elif last_month is not None and month < last_month:
                if last_month >= 11 and month <= 2:
                    cur_year += 1                 
                else:
                    current = None
                    continue
            last_month = month
            try:
                when = datetime(cur_year, month, day)
            except ValueError:
                current = None                     
                continue
            current = {"Date": when, "parts": [], "Amount_abs": None, "Balance": None}
        else:
            if current is None:
                continue                          
            text = line

        money = MONEY_RE.findall(text)
        if len(money) >= 2 and current["Amount_abs"] is None:
            current["Balance"] = _to_float(money[-1])
            current["Amount_abs"] = abs(_to_float(money[-2]))
            text = MONEY_RE.sub("", text)

        text = text.strip()
        if text:
            current["parts"].append(text)

        if current["Amount_abs"] is not None:     
            records.append({
                "Date": current["Date"],
                "Amount_abs": current["Amount_abs"],
                "Description": re.sub(r"\s+", " ", " ".join(current["parts"])).strip(),
                "Balance": current["Balance"],
            })
            current = None

    if not records:
        return None

    df = pd.DataFrame(records)

    prev = opening_balance
    signed, bad_rows = [], []
    for i, row in df.iterrows():
        if prev is None:
            sign = -1                              # no opening balance: assume debit
        else:
            delta = round(row["Balance"] - prev, 2)
            if abs(abs(delta) - row["Amount_abs"]) > 0.01:
                bad_rows.append(i)                 
            sign = 1 if delta > 0 else -1
        signed.append(sign * row["Amount_abs"])
        prev = row["Balance"]
    df["Amount"] = signed

    if bad_rows:
        warnings.warn(f"{len(bad_rows)} row(s) don't reconcile with the running balance "
                      f"(rows {bad_rows[:10]}...). A transaction may have been missed.")
    if closing_balances and not any(abs(df["Balance"].iloc[-1] - c) < 0.01 for c in closing_balances):
        warnings.warn(f"Last parsed balance {df['Balance'].iloc[-1]:.2f} doesn't match the "
                      f"statement's closing balance {closing_balances}. Rows may be missing "
                      f"from the end of the statement.")
    df.attrs["unreconciled_rows"] = bad_rows
    return df[["Date", "Amount", "Description", "Balance"]]


def parse_pdf(source) -> pd.DataFrame:
    """source: a file path OR a file-like object (e.g. io.BytesIO from Streamlit)."""
    with pdfplumber.open(source) as pdf:
        for tol in X_TOLERANCES:
            lines = []
            for page in pdf.pages:
                lines.extend((page.extract_text(x_tolerance=tol) or "").splitlines())
            if not lines:
                raise ValueError("No text found - this PDF is probably scanned/image-based "
                                 "(needs OCR).")
            df = _parse_lines(lines)
            if df is not None:
                return df

    raise ValueError("No transaction lines recognised. Print page.extract_text() for one page "
                     "and compare it with the DATE_RE / MONEY_RE patterns at the top of this file.")