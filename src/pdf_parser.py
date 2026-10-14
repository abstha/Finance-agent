"""
pdf_parser.py - turn a text-based bank statement PDF into the same DataFrame
that load_data() produces from a CSV: Date, Amount, Description, Balance.

Layout this is written for (CommBank-style):

    01 Mar 2026 OPENING BALANCE                              $71.32 CR
    01 Mar Debit Excess Interest                  0.01       $71.31 CR
    03 Mar TRANSPORTFORNSW TAP SYDNEY AUS
           Card xx1835
           Value Date: 27/02/2026                10.82       $60.49 CR

How it works:
  1. A line that STARTS with a date ("03 Mar") opens a new transaction block.
  2. Following lines without a date are part of the same block (wrapped
     description, "Card xx1835", "Value Date: ...").
  3. The first line in the block holding TWO money values gives us
     (amount, balance), and closes the block. Because the block is closed,
     the private header/footer text around the table is never swallowed.
  4. The SIGN of each amount comes from how the running balance moved, not from
     which column (Debit/Credit) the number sat in.
  5. Dates have no year, so the year is taken from the OPENING BALANCE line and
     rolled forward when the month wraps (Dec -> Jan).
"""
import re
import warnings
from datetime import datetime

import pandas as pd
import pdfplumber

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# "03 Mar", "03 Mar 2026", "03 March" at the START of a line, then the rest.
TEXT_DATE_RE = re.compile(
    r"^\s*(?P<day>\d{1,2})\s*"
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])"
    r"(?:\s*(?P<year>20\d{2})(?!\d))?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
# "03/03/2026" or "03/03/26" at the start of a line (other banks)
SLASH_DATE_RE = re.compile(
    r"^\s*(?P<day>\d{1,2})/(?P<mon>\d{1,2})/(?P<year>\d{2,4})\s+(?P<rest>.*)$"
)

# 1,234.56   $1,234.56   (45.00)   -45.00   45.00 CR   $10.63 DR
MONEY_RE = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?(?:\s?(?:CR|DR)\b)?", re.IGNORECASE)

# matched against the line with ALL spaces removed, so "ClosingBalance" and "Closing Balance" both hit
STOP_HINTS = ("closingbalance", "balancebroughtforward", "balancecarriedforward")

# pdfplumber merges two words when the gap between them is < x_tolerance points. Many bank PDFs
# have no real space characters (only small gaps), so the default (3) glues words together
# ("01Mar DebitExcessInterest"). Try tighter settings first; the parser also tolerates
# merged text, but descriptions read far better (and categorize better) with spaces restored.
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

    cur_year = None        # set from the OPENING BALANCE line when possible
    last_month = None      # to detect Dec -> Jan rollover
    opening_balance = None
    closing_balances = []  # used for a final integrity check
    records = []
    current = None         # the open transaction block (None = nothing open)

    for line in lines:
        squash = line.lower().replace(" ", "")

        # --- opening balance: gives us the starting balance AND the year ---
        if "openingbalance" in squash:
            money = MONEY_RE.findall(line)
            if money and opening_balance is None:
                opening_balance = _to_float(money[-1])
            d = _match_date(line)
            if d and d[2] and cur_year is None:
                cur_year = d[2]
            current = None
            continue

        # --- end of the table / page furniture: close any open block ---
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
                    cur_year += 1                  # genuine year wrap: Nov/Dec -> Jan/Feb
                else:
                    # Dates go backwards mid-year: this is a footnote / summary-table line
                    # that merely starts with a date-like token, not a transaction.
                    current = None
                    continue
            last_month = month
            try:
                when = datetime(cur_year, month, day)
            except ValueError:
                current = None                     # not a real date (e.g. "31 Feb")
                continue
            current = {"Date": when, "parts": [], "Amount_abs": None, "Balance": None}
        else:
            if current is None:
                continue                           # header / footer / private details
            text = line

        money = MONEY_RE.findall(text)
        if len(money) >= 2 and current["Amount_abs"] is None:
            current["Balance"] = _to_float(money[-1])
            current["Amount_abs"] = abs(_to_float(money[-2]))
            text = MONEY_RE.sub("", text)

        text = text.strip()
        if text:
            current["parts"].append(text)

        if current["Amount_abs"] is not None:      # block complete -> close it
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

    # --- sign each amount from the balance movement, and sanity-check it ---
    prev = opening_balance
    signed, bad_rows = [], []
    for i, row in df.iterrows():
        if prev is None:
            sign = -1                              # no opening balance: assume debit
        else:
            delta = round(row["Balance"] - prev, 2)
            if abs(abs(delta) - row["Amount_abs"]) > 0.01:
                bad_rows.append(i)                 # a transaction was probably missed
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