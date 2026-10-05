"""
BANK STATEMENT EXTRACTION & TRANSACTION INTELLIGENCE ENGINE

Keeps the original StatementPipeline style and function names, but extends the
old extraction/classification flow into:

inspect
-> boundaries
-> account
-> layout
-> transactions
-> validate
-> classify
-> identify_counterparties
-> resolve_counterparty_entities
-> detect_transaction_relationships
-> detect_duplicates
-> cluster_and_aggregate
-> detect_anomalies
-> generate_ca_review_queue
-> generate_final_summary

No LangGraph is required for the normal path. The pipeline is deterministic
and easier to debug. The LLM is used only where it adds value: extraction,
classification, counterparty understanding, and accounting-purpose reasoning.

pip install langchain pydantic
"""

import concurrent.futures
import difflib
import functools
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from apps.common.llms import get_ollama_instance

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
# Controlled values
# ----------------------------------------------------------------------------------------

PAYMENT_MODES = {
    "UPI",
    "NEFT",
    "RTGS",
    "IMPS",
    "BANK_TRANSFER",
    "CHEQUE",
    "CASH",
    "ATM",
    "DEBIT_CARD",
    "CREDIT_CARD",
    "POS",
    "NACH",
    "ECS",
    "INTEREST",
    "BANK_CHARGE",
    "REVERSAL",
    "REFUND",
    "UNKNOWN",
}

RELATIONSHIPS = {
    "PAYMENT",
    "RECEIPT",
    "REFUND",
    "REVERSAL",
    "TRANSFER",
    "REIMBURSEMENT",
    "LOAN_DISBURSEMENT",
    "LOAN_REPAYMENT",
    "INTEREST",
    "CHARGE",
    "TAX_PAYMENT",
    "SALARY",
    "PURCHASE",
    "SALE_RECEIPT",
    "UNKNOWN",
}

COUNTERPARTY_TYPES = {
    "PERSON",
    "BUSINESS",
    "BANK",
    "GOVERNMENT",
    "PLATFORM",
    "UNKNOWN",
}

ACCOUNTING_CATEGORIES = {
    "SALES_RECEIPT",
    "PURCHASE",
    "RAW_MATERIAL",
    "INVENTORY",
    "SALARY",
    "RENT",
    "PROFESSIONAL_FEES",
    "TRAVEL",
    "TELECOMMUNICATION",
    "UTILITIES",
    "SOFTWARE_SUBSCRIPTION",
    "ADVERTISING",
    "BANK_CHARGES",
    "INTEREST",
    "LOAN_REPAYMENT",
    "LOAN_RECEIPT",
    "TAX_PAYMENT",
    "GST_PAYMENT",
    "TDS_PAYMENT",
    "GOVERNMENT_PAYMENT",
    "INSURANCE",
    "REFUND",
    "REIMBURSEMENT",
    "INTERNAL_TRANSFER",
    "PERSONAL_EXPENSE",
    "CAPITAL_EXPENDITURE",
    "INVESTMENT",
    "DIVIDEND",
    "INTEREST_INCOME",
    "CASH_DEPOSIT",
    "CASH_WITHDRAWAL",
    "UNKNOWN",
}

ALLOWED_GROUP_FIELDS = (
    "direction",
    "category",
    "counterparty",
    "month",
    "payment_mode",
    "relationship",
)


# ----------------------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------------------


class MainAccount(BaseModel):
    bank_name: Optional[str] = None
    account_holder_name: Optional[str] = None
    account_holder_type: Optional[str] = None
    account_holder_address: Optional[str] = None
    account_number: Optional[str] = None
    masked_account: Optional[str] = None
    branch: Optional[str] = None
    branch_address: Optional[str] = None
    ifsc: Optional[str] = None
    micr: Optional[str] = None
    branch_phone: Optional[str] = None
    account_type: Optional[str] = None
    customer_id: Optional[str] = None
    statement_number: Optional[str] = None
    currency: Optional[str] = None
    statement_date: Optional[str] = None
    statement_period: Optional[str] = None
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    nominee_registered: Optional[bool] = None
    registered_email: Optional[str] = None
    registered_phone: Optional[str] = None
    pan: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None


class BoundaryGuess(BaseModel):
    first_transaction_line: int


class LayoutInfo(BaseModel):
    column_names: list[str] = Field(default_factory=list)
    direction_style: str
    date_format: Optional[str] = None
    row_layout: str
    has_balance_column: bool = False
    has_reference_column: bool = False


class RawTransaction(BaseModel):
    date: Optional[str] = None
    value_date: Optional[str] = None
    date_original: Optional[str] = None
    description: str
    amount: float
    direction: str
    balance: Optional[float] = None
    currency: Optional[str] = None
    reference_number: Optional[str] = None
    cheque_number: Optional[str] = None
    source_page: Optional[str] = None
    source_row: Optional[str] = None
    source_sheet: Optional[str] = None
    extraction_confidence: Optional[float] = None


class TransactionExtraction(BaseModel):
    transactions: list[RawTransaction]


class Transaction(RawTransaction):
    transaction_id: str


class ClassifiedTransaction(BaseModel):
    transaction_id: str
    category: str = "UNKNOWN"
    subcategory: Optional[str] = None
    counterparty: Optional[str] = None
    counterparty_type: str = "UNKNOWN"
    payment_mode: str = "UNKNOWN"
    relationship: str = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0, le=1)
    classification_reason: Optional[str] = None


class ClassificationBatch(BaseModel):
    items: list[ClassifiedTransaction]


class CategoryMergeItem(BaseModel):
    label: str
    canonical: str


class CategoryMerge(BaseModel):
    items: list[CategoryMergeItem]


class EntityResolution(BaseModel):
    transaction_id: str
    counterparty_name_normalized: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: list[str] = Field(default_factory=list)


class EntityResolutionBatch(BaseModel):
    items: list[EntityResolution]


class TransactionCluster(BaseModel):
    cluster_id: str
    account_name: Optional[str] = None
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    cluster_type: str
    cluster_name: Optional[str] = None
    transaction_type: str
    category: Optional[str] = None
    counterparty: Optional[str] = None
    month: Optional[str] = None
    payment_mode: Optional[str] = None
    relationship: Optional[str] = None
    transaction_count: int
    credit_total: float
    debit_total: float
    total_amount: float
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    recurring: bool = False
    confidence: float = 0.0
    transaction_ids: list[str] = Field(default_factory=list)


class CounterpartySummary(BaseModel):
    counterparty_id: Optional[str] = None
    counterparty: str
    transaction_count: int
    total_received: float = 0.0
    total_paid: float = 0.0
    net_amount: float = 0.0
    first_transaction: Optional[str] = None
    last_transaction: Optional[str] = None
    payment_modes: list[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    confidence: float = 0.0


class ReviewItem(BaseModel):
    transaction_id: Optional[str] = None
    issue_type: str
    reason: str
    severity: str = "MEDIUM"
    recommended_review: Optional[str] = None


class ValidationResult(BaseModel):
    transaction_count: int = 0
    credit_total: float = 0.0
    debit_total: float = 0.0
    balance_consistency_percentage: float = 0.0
    date_consistency_percentage: float = 0.0
    duplicate_row_count: int = 0
    missing_data_count: int = 0
    validation_status: str = "WARNING"
    issues: list[str] = Field(default_factory=list)


class StatementResult(BaseModel):
    account: MainAccount
    group_by: list[str]
    transaction_count: int
    total_credit: float
    total_debit: float
    net_movement: float
    warnings: list[str] = Field(default_factory=list)

    validation: ValidationResult = Field(default_factory=ValidationResult)

    clusters: list[TransactionCluster] = Field(default_factory=list)

    money_received: list[CounterpartySummary] = Field(default_factory=list)
    money_paid: list[CounterpartySummary] = Field(default_factory=list)

    accounting_summary: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    payment_mode_summary: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    monthly_summary: dict[str, dict[str, float | int]] = Field(default_factory=dict)

    review_queue: list[ReviewItem] = Field(default_factory=list)
    anomalies: list[ReviewItem] = Field(default_factory=list)

    data_quality: dict[str, Any] = Field(default_factory=dict)

    transactions: Optional[list[dict[str, Any]]] = None
    summary: Optional[str] = None


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

DATE_RE = re.compile(r"^\s*(\d{1,2}[-/. ]([A-Za-z]{3,9}|\d{1,2})[-/. ,]+\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})")
AMOUNT_RE = re.compile(r"\d[\d,]*\.?\d{0,2}")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

UPI_RE = re.compile(r"(?i)(?:upi[/:\-])?([A-Za-z0-9._-]+@[A-Za-z0-9._-]+)")
UTR_RE = re.compile(r"(?i)\b(?:UTR|RRN|REF|REFERENCE)[\s:/-]*([A-Za-z0-9]+)")
CHEQUE_RE = re.compile(r"(?i)\b(?:CHEQUE|CHQ|CHK)[\s:/#-]*(\d+)")

PAYMENT_MODE_PATTERNS = [
    ("UPI", r"\bUPI\b|@\w+"),
    ("NEFT", r"\bNEFT\b"),
    ("RTGS", r"\bRTGS\b"),
    ("IMPS", r"\bIMPS\b"),
    ("NACH", r"\bNACH\b"),
    ("ECS", r"\bECS\b"),
    ("ATM", r"\bATM\b"),
    ("POS", r"\bPOS\b"),
    ("DEBIT_CARD", r"\bDEBIT\s*CARD\b"),
    ("CREDIT_CARD", r"\bCREDIT\s*CARD\b"),
    ("CHEQUE", r"\bCHEQUE\b|\bCHQ\b|\bCHK\b"),
    ("CASH", r"\bCASH\b"),
]

BANK_WORDS = {
    "BANK",
    "HDFC",
    "ICICI",
    "AXIS",
    "SBI",
    "KOTAK",
    "IDFC",
    "YES BANK",
    "INDUSIND",
    "PNB",
    "BOB",
    "CANARA",
    "UNION BANK",
}

REVIEW_LOW_CONFIDENCE = 0.60


def chunk_lines(text: str, max_chars: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for line in text.splitlines():
        if cur and size + len(line) > max_chars and (DATE_RE.match(line) or size > max_chars * 1.5):
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    value = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()
    return value or None


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    value = re.sub(r"[^A-Za-z0-9&.\-@ ]+", " ", name).upper()
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def parse_amount(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return abs(float(value))
    text = str(value).replace(",", "").strip()
    try:
        return abs(float(text))
    except ValueError:
        return None


def canonicalize_names(names: list[Optional[str]]) -> dict[str, str]:
    """
    Conservative entity merge.

    Exact normalized names are always merged.
    Similar names are merged only when the similarity is high.
    """
    values = Counter(n for n in names if n)
    canonical: list[str] = []
    mapping: dict[str, str] = {}

    for name, _ in values.most_common():
        exact = next((c for c in canonical if c == name), None)
        if exact:
            mapping[name] = exact
            continue

        match = difflib.get_close_matches(name, canonical, n=1, cutoff=0.92)
        if match:
            mapping[name] = match[0]
        else:
            canonical.append(name)
            mapping[name] = name

    return mapping


def extract_payment_mode(description: str) -> str:
    text = description.upper()
    for mode, pattern in PAYMENT_MODE_PATTERNS:
        if re.search(pattern, text):
            return mode

    if "CHARGE" in text or "FEE" in text:
        return "BANK_CHARGE"
    if "INTEREST" in text:
        return "INTEREST"
    if "REVERS" in text:
        return "REVERSAL"
    if "REFUND" in text:
        return "REFUND"

    return "UNKNOWN"


def extract_reference(description: str) -> Optional[str]:
    match = UTR_RE.search(description)
    if match:
        return match.group(1)
    return None


def extract_upi(description: str) -> Optional[str]:
    match = UPI_RE.search(description)
    if match:
        return match.group(1)
    return None


def extract_cheque(description: str) -> Optional[str]:
    match = CHEQUE_RE.search(description)
    if match:
        return match.group(1)
    return None


def safe_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if ISO_RE.match(value):
        return value
    return None


def month_of(value: Optional[str]) -> Optional[str]:
    return value[:7] if value and ISO_RE.match(value) else None


def is_same_or_near_amount(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(a - b) <= tolerance


def step(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                msg = fn(self, *args, **kwargs)
            except Exception as exc:
                logger.exception("step %s failed", name)
                return f"ERROR: {name} failed: {exc}"
            if isinstance(msg, str) and not msg.startswith("ERROR"):
                self.done.add(name)
            return msg

        return wrapper

    return deco


# ----------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------


class StatementPipeline:
    def __init__(
        self,
        raw_text: str,
        llm,
        group_by: Optional[list[str]] = None,
        include_transactions: bool = False,
        chunk_size: int = 3500,
        batch_size: int = 25,
        max_workers: int = 2,
    ):
        self.raw_text = raw_text
        self.lines = raw_text.splitlines()
        self.llm = llm

        extra = [
            field for field in (group_by or ["category"]) if field in ALLOWED_GROUP_FIELDS and field != "direction"
        ]
        self.group_by = ["direction", *dict.fromkeys(extra)]

        self.include_transactions = include_transactions
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_workers = max_workers

        self.first_txn: Optional[int] = None
        self.header_text = ""
        self.txn_text = ""

        self.account: Optional[MainAccount] = None
        self.layout: Optional[LayoutInfo] = None
        self.transactions: list[Transaction] = []
        self.classified: dict[str, ClassifiedTransaction] = {}

        self.validation = ValidationResult()
        self.clusters: list[TransactionCluster] = []
        self.money_received: list[CounterpartySummary] = []
        self.money_paid: list[CounterpartySummary] = []

        self.review_queue: list[ReviewItem] = []
        self.anomalies: list[ReviewItem] = []

        self.warn: dict[str, list[str]] = {}
        self.done: set[str] = set()
        self.result: Optional[StatementResult] = None
        self.final_summary: Optional[str] = None

    # ------------------------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------------------------

    def _ask(self, schema, prompt: str, retries: int = 2):
        structured = self.llm.with_structured_output(schema)
        last_error = None

        for _ in range(retries + 1):
            try:
                output = structured.invoke(prompt)
                if output is not None:
                    return output
            except Exception as exc:
                last_error = exc
                logger.warning("LLM call for %s failed: %s", schema.__name__, exc)

        raise RuntimeError(f"LLM could not produce {schema.__name__}: {last_error}")

    # ------------------------------------------------------------------------------------
    # 0. inspect
    # ------------------------------------------------------------------------------------

    def inspect(self) -> str:
        preview = "\n".join(line[:150] for line in self.lines[:8])
        return f"chars={len(self.raw_text)} lines={len(self.lines)}\n" f"PREVIEW:\n{preview}"

    @step("inspect")
    def inspect_statement(self) -> str:
        return self.inspect()

    # ------------------------------------------------------------------------------------
    # 1. boundaries
    # ------------------------------------------------------------------------------------

    def _heuristic_first_txn(self) -> Optional[int]:
        for i, line in enumerate(self.lines):
            if DATE_RE.match(line):
                amount_lines = sum(bool(AMOUNT_RE.search(item)) for item in self.lines[i : i + 12])
                if amount_lines >= 2:
                    return i
        return None

    @step("boundaries")
    def detect_boundaries(self) -> str:
        numbered = "\n".join(f"{i}: {line[:180]}" for i, line in enumerate(self.lines[:250]))

        index = None
        method = "llm"

        try:
            index = self._ask(
                BoundaryGuess,
                """
You are locating the transaction table in a bank statement.

Find the first line of the first genuine money-movement transaction.
Do not select account details, column headers, opening balance, totals,
closing balance, disclaimers, or page metadata.

Return the exact 0-based line number shown below.

""" + numbered,
            ).first_transaction_line
        except RuntimeError:
            pass

        valid = (
            index is not None
            and 0 <= index < len(self.lines)
            and any(DATE_RE.match(line) for line in self.lines[max(0, index - 1) : index + 4])
        )

        if not valid:
            index = self._heuristic_first_txn()
            method = "heuristic"

        if index is None:
            return "ERROR: could not find where the transactions start."

        self.first_txn = index
        self.header_text = "\n".join(self.lines[:index])
        self.txn_text = "\n".join(self.lines[index:])

        return f"OK method={method} header_lines={index} " f"transaction_lines={len(self.lines) - index}"

    # ------------------------------------------------------------------------------------
    # 2. account
    # ------------------------------------------------------------------------------------

    @step("account")
    def extract_account(self) -> str:
        if not self.header_text.strip():
            return "ERROR: header is empty. Run detect_boundaries first."

        self.account = self._ask(
            MainAccount,
            """
Extract account-level information from this bank statement header.

Rules:
- Use null when the value is not printed.
- Never reconstruct masked account numbers.
- Never infer PAN, IFSC, account type, holder name, dates, or balances.
- Preserve source values as closely as possible.
- Extract opening/closing balances only when explicitly present.

HEADER:
""" + self.header_text[:16000],
        )

        account = self.account
        return (
            f"OK bank={account.bank_name} "
            f"holder={account.account_holder_name} "
            f"account={account.masked_account or account.account_number} "
            f"type={account.account_type}"
        )

    # ------------------------------------------------------------------------------------
    # 3. layout
    # ------------------------------------------------------------------------------------

    @step("layout")
    def discover_layout(self) -> str:
        if self.first_txn is None:
            return "ERROR: run detect_boundaries first."

        sample = "\n".join(self.lines[max(0, self.first_txn - 30) : self.first_txn + 55])

        self.layout = self._ask(
            LayoutInfo,
            """
Study this bank statement excerpt and identify its actual transaction-table
layout.

Do not assume bank-specific columns.
Determine:
- printed column names
- how debit/credit is represented
- date format
- row wrapping
- whether balance and reference columns exist

Do not invent a column that is not evidenced by the source.

EXCERPT:
""" + sample,
        )

        return (
            f"OK columns={self.layout.column_names} "
            f"direction_style={self.layout.direction_style!r} "
            f"row_layout={self.layout.row_layout!r}"
        )

    # ------------------------------------------------------------------------------------
    # 4. raw transactions
    # ------------------------------------------------------------------------------------

    def _txn_prompt(self, chunk: str) -> str:
        return (
            """
Extract every genuine bank transaction from the raw statement chunk.

Rules:
- One item = one real bank transaction.
- Never combine separate transactions.
- Join wrapped narration lines belonging to the same transaction.
- Ignore page headers/footers, repeated headers, totals, summaries,
  opening/closing balances and disclaimers.
- amount is always positive.
- direction is exactly CREDIT or DEBIT.
- Determine direction from the discovered table layout, not from a generic
  assumption that positive means credit.
- Preserve original narration.
- Preserve original date text in date_original.
- Normalize transaction_date to YYYY-MM-DD when evidence allows.
- Use null rather than guessing.
- balance is the printed running balance after the transaction, otherwise null.
- source page/row/sheet may remain null if unavailable.

DISCOVERED LAYOUT:
"""
            + self.layout.model_dump_json()
            + """

RAW CHUNK:
"""
            + chunk
        )

    @step("transactions")
    def extract_transactions(self, chunk_size: Optional[int] = None) -> str:
        if not self.txn_text:
            return "ERROR: no transaction text. Run detect_boundaries first."

        if self.layout is None:
            layout_result = self.discover_layout()
            if layout_result.startswith("ERROR"):
                return "ERROR: layout discovery failed."

        chunks = chunk_lines(self.txn_text, chunk_size or self.chunk_size)

        def extract_one(chunk: str):
            try:
                return self._ask(
                    TransactionExtraction,
                    self._txn_prompt(chunk),
                ).transactions
            except RuntimeError:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(extract_one, chunks))

        failed = [index for index, result in enumerate(results) if result is None]

        if failed:
            return f"ERROR: chunks {failed} of {len(chunks)} failed. " f"Retry with a smaller chunk_size (e.g. 2000)."

        self.transactions = []
        self.classified = {}
        self.result = None

        self.done -= {
            "validate",
            "classify",
            "counterparties",
            "entities",
            "relationships",
            "duplicates",
            "cluster",
            "anomalies",
            "review",
            "summary",
        }

        for rows in results:
            for row in rows:
                direction = row.direction.strip().upper()
                row.direction = "CREDIT" if direction.startswith(("CR", "C")) else "DEBIT"

                row.amount = abs(row.amount)

                if row.extraction_confidence is None:
                    row.extraction_confidence = 0.8

                self.transactions.append(
                    Transaction(
                        transaction_id=f"tx_{len(self.transactions) + 1:06d}",
                        **row.model_dump(),
                    )
                )

        if not self.transactions:
            return "ERROR: no transactions were extracted."

        return f"OK chunks={len(chunks)} " f"transactions={len(self.transactions)}"

    # ------------------------------------------------------------------------------------
    # 5. validation
    # ------------------------------------------------------------------------------------

    @step("validate")
    def validate_transactions(self) -> str:
        tx = self.transactions

        if not tx:
            return "ERROR: no transactions. Run extract_transactions first."

        credit_total = round(sum(t.amount for t in tx if t.direction == "CREDIT"), 2)
        debit_total = round(sum(t.amount for t in tx if t.direction == "DEBIT"), 2)

        dated = [t for t in tx if t.date and ISO_RE.match(t.date)]

        date_consistency = 100 * len(dated) / len(tx) if tx else 0

        # Determine whether extracted transactions are ascending or descending
        # by comparing balance consistency in both directions.
        def score(descending: bool):
            pairs = []
            ordered = tx

            for a, b in zip(ordered, ordered[1:]):
                if a.balance is None or b.balance is None:
                    continue
                pairs.append((b, a) if descending else (a, b))

            ok = 0
            for previous, current in pairs:
                delta = current.balance - previous.balance
                expected = current.amount if current.direction == "CREDIT" else -current.amount
                if abs(delta - expected) < 0.01:
                    ok += 1

            return ok, len(pairs)

        ascending = score(False)
        descending = score(True)

        use_descending = descending[0] > ascending[0]
        balance_ok, balance_total = descending if use_descending else ascending

        corrected = 0

        # Only correct direction when the running balance gives very strong
        # evidence for that individual row.
        ordered_pairs = []
        for a, b in zip(tx, tx[1:]):
            if a.balance is None or b.balance is None:
                continue
            ordered_pairs.append((b, a) if use_descending else (a, b))

        for previous, current in ordered_pairs:
            delta = current.balance - previous.balance

            if abs(abs(delta) - current.amount) < 0.01:
                true_direction = "CREDIT" if delta > 0 else "DEBIT"

                if current.direction != true_direction:
                    current.direction = true_direction
                    corrected += 1

        balance_percentage = 100 * balance_ok / balance_total if balance_total else 0

        duplicate_count = self._count_exact_duplicate_rows()

        missing_data = sum(
            1 for t in tx if not t.description or t.amount is None or t.direction not in {"CREDIT", "DEBIT"}
        )

        issues = []

        if balance_total and balance_percentage < 90:
            issues.append(f"Running balance consistency is only " f"{balance_percentage:.1f}%.")

        if date_consistency < 95:
            issues.append(f"{len(tx) - len(dated)} transactions have missing/invalid dates.")

        if duplicate_count:
            issues.append(f"{duplicate_count} exact duplicate transaction rows detected.")

        if missing_data:
            issues.append(f"{missing_data} transactions have missing required data.")

        if balance_percentage >= 90 and date_consistency >= 95:
            status = "PASS"
        elif balance_percentage >= 70:
            status = "WARNING"
        else:
            status = "FAIL"

        self.validation = ValidationResult(
            transaction_count=len(tx),
            credit_total=credit_total,
            debit_total=debit_total,
            balance_consistency_percentage=round(balance_percentage, 2),
            date_consistency_percentage=round(date_consistency, 2),
            duplicate_row_count=duplicate_count,
            missing_data_count=missing_data,
            validation_status=status,
            issues=issues,
        )

        self.warn["validate"] = issues

        return (
            f"OK status={status} "
            f"transactions={len(tx)} "
            f"credits={credit_total:,.2f} "
            f"debits={debit_total:,.2f} "
            f"balance_consistency={balance_percentage:.1f}% "
            f"directions_corrected={corrected}"
        )

    def _count_exact_duplicate_rows(self) -> int:
        seen = set()
        duplicates = 0

        for tx in self.transactions:
            key = (
                tx.date,
                round(tx.amount, 2),
                tx.direction,
                normalize_name(tx.description),
                tx.reference_number,
            )

            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

        return duplicates

    # ------------------------------------------------------------------------------------
    # 6. classification
    # ------------------------------------------------------------------------------------

    def _cls_prompt(
        self,
        batch: list[Transaction],
        labels_in_use: list[str],
    ) -> str:
        rows = "\n".join(
            (f"{t.transaction_id} | {t.date} | {t.direction} | " f"{t.amount:.2f} | {t.description[:300]}")
            for t in batch
        )

        categories = ", ".join(sorted(ACCOUNTING_CATEGORIES))
        existing = ", ".join(labels_in_use[:80]) or "(none yet)"

        return f"""
Classify every bank transaction below.

The objective is accounting/business-purpose classification, not merely
debit/credit classification.

Controlled category taxonomy:
{categories}

Rules:
- Reuse a category already in use when it genuinely means the same thing.
- If none fits, use UNKNOWN rather than forcing a category.
- A CREDIT is not automatically income.
- A DEBIT is not automatically an expense.
- Identify payer for CREDIT and payee/merchant for DEBIT when evidence exists.
- Do not invent a counterparty.
- Extract UPI/payment mode only when the narration supports it.
- relationship must be one of:
  {sorted(RELATIONSHIPS)}
- counterparty_type must be one of:
  {sorted(COUNTERPARTY_TYPES)}
- confidence must reflect evidence.
- Low-confidence or ambiguous transactions should remain UNKNOWN and later
  be sent for review.
- Return exactly one item for every transaction ID.

Existing labels in this run:
{existing}

Transactions:
id | date | direction | amount | description
{rows}
"""

    @step("classify")
    def classify_transactions(
        self,
        batch_size: Optional[int] = None,
    ) -> str:
        if not self.transactions:
            return "ERROR: no transactions. Run extract_transactions first."

        bs = batch_size or self.batch_size

        labels_in_use: list[str] = []
        raw: dict[str, ClassifiedTransaction] = {}

        for start in range(0, len(self.transactions), bs):
            batch = self.transactions[start : start + bs]
            valid_ids = {t.transaction_id for t in batch}

            try:
                output = self._ask(
                    ClassificationBatch,
                    self._cls_prompt(batch, labels_in_use),
                )
            except RuntimeError:
                return (
                    f"ERROR: classification batch starting at {start} failed. "
                    f"Retry with a smaller batch_size (e.g. 10)."
                )

            for item in output.items:
                if item.transaction_id not in valid_ids:
                    continue

                item.category = normalize_label(item.category) or "UNKNOWN"

                if item.category not in ACCOUNTING_CATEGORIES:
                    item.category = "UNKNOWN"

                item.counterparty = normalize_name(item.counterparty)

                item.counterparty_type = item.counterparty_type.upper() if item.counterparty_type else "UNKNOWN"

                if item.counterparty_type not in COUNTERPARTY_TYPES:
                    item.counterparty_type = "UNKNOWN"

                item.payment_mode = item.payment_mode.upper()
                if item.payment_mode not in PAYMENT_MODES:
                    item.payment_mode = extract_payment_mode(
                        next(
                            (t.description for t in batch if t.transaction_id == item.transaction_id),
                            "",
                        )
                    )

                item.relationship = item.relationship.upper()
                if item.relationship not in RELATIONSHIPS:
                    item.relationship = "UNKNOWN"

                raw[item.transaction_id] = item

                if item.category not in labels_in_use:
                    labels_in_use.append(item.category)

        missing = 0

        self.classified = {}

        names = canonicalize_names([item.counterparty for item in raw.values()])

        for tx in self.transactions:
            item = raw.get(tx.transaction_id)

            if item is None:
                missing += 1
                item = ClassifiedTransaction(
                    transaction_id=tx.transaction_id,
                    category="UNKNOWN",
                    counterparty=None,
                    confidence=0.0,
                )
            elif item.counterparty:
                item.counterparty = names.get(
                    item.counterparty,
                    item.counterparty,
                )

            # Deterministic extraction is stronger than an LLM omission.
            if item.payment_mode == "UNKNOWN":
                item.payment_mode = extract_payment_mode(tx.description)

            if not item.counterparty:
                extracted_upi = extract_upi(tx.description)
                if extracted_upi:
                    item.counterparty = normalize_name(extracted_upi)
                    item.confidence = min(
                        max(item.confidence, 0.70),
                        1.0,
                    )

            if item.relationship == "UNKNOWN":
                item.relationship = self._infer_relationship(
                    tx,
                    item,
                )

            self.classified[tx.transaction_id] = item

        self._consolidate_categories()

        low = sum(1 for item in self.classified.values() if item.confidence < REVIEW_LOW_CONFIDENCE)

        warnings = []

        if missing:
            warnings.append(f"{missing} transactions were not classified by the model.")

        if low:
            warnings.append(f"{low} transactions have low classification confidence.")

        self.warn["classify"] = warnings

        return f"OK classified={len(self.classified)} " f"missing={missing} " f"low_confidence={low}"

    def _infer_relationship(
        self,
        tx: Transaction,
        item: ClassifiedTransaction,
    ) -> str:
        text = tx.description.upper()

        if "REFUND" in text:
            return "REFUND"

        if "REVERS" in text:
            return "REVERSAL"

        if "INTEREST" in text:
            return "INTEREST"

        if "CHARGE" in text or "FEE" in text:
            return "CHARGE"

        if item.category in {"SALARY"}:
            return "SALARY"

        if item.category in {"GST_PAYMENT", "TDS_PAYMENT", "TAX_PAYMENT"}:
            return "TAX_PAYMENT"

        if item.category == "LOAN_RECEIPT":
            return "LOAN_DISBURSEMENT"

        if item.category == "LOAN_REPAYMENT":
            return "LOAN_REPAYMENT"

        if item.category == "INTERNAL_TRANSFER":
            return "TRANSFER"

        if tx.direction == "CREDIT":
            return "RECEIPT"

        if tx.direction == "DEBIT":
            return "PAYMENT"

        return "UNKNOWN"

    def _consolidate_categories(self):
        counts = Counter(item.category for item in self.classified.values() if item.category != "UNKNOWN")

        if len(counts) < 3:
            return

        listing = "\n".join(f"{label} ({count})" for label, count in counts.most_common(100))

        try:
            output = self._ask(
                CategoryMerge,
                """
Merge only categories that clearly represent the same accounting purpose.

Do not merge unrelated categories.
Keep UNKNOWN separate.
Every input label should be returned exactly once.

LABELS:
""" + listing,
            )
        except RuntimeError:
            self.warn["merge"] = ["Category consolidation was skipped."]
            return

        mapping = {}

        for item in output.items:
            source = normalize_label(item.label)
            canonical = normalize_label(item.canonical)

            if source in counts and canonical in ACCOUNTING_CATEGORIES:
                mapping[source] = canonical

        for item in self.classified.values():
            item.category = mapping.get(
                item.category,
                item.category,
            )

    # ------------------------------------------------------------------------------------
    # 7. counterparty identification
    # ------------------------------------------------------------------------------------

    @step("counterparties")
    def identify_counterparties(self) -> str:
        if not self.transactions or len(self.classified) != len(self.transactions):
            return "ERROR: transactions must be classified before " "counterparty identification."

        identified = 0

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]

            if item.counterparty:
                identified += 1
                continue

            description = tx.description

            # Conservative extraction from common bank narration patterns.
            parts = [
                p.strip()
                for p in re.split(
                    r"[/|:\-]+",
                    description,
                )
                if p.strip()
            ]

            candidates = []

            for part in parts:
                normalized = normalize_name(part)

                if not normalized:
                    continue

                if normalized in {
                    "UPI",
                    "NEFT",
                    "RTGS",
                    "IMPS",
                    "TRANSFER",
                    "PAYMENT",
                    "BANK",
                    "CASH",
                    "ATM",
                    "POS",
                    "REF",
                    "REFERENCE",
                    "CHARGE",
                }:
                    continue

                if re.fullmatch(r"\d+", normalized):
                    continue

                if normalized in BANK_WORDS:
                    continue

                if "@" in normalized:
                    candidates.append(normalized)

                elif len(normalized.split()) >= 2:
                    candidates.append(normalized)

            if candidates:
                item.counterparty = candidates[0]
                item.confidence = max(
                    item.confidence,
                    0.55,
                )
                identified += 1

        return f"OK identified={identified} " f"total={len(self.transactions)}"

    # ------------------------------------------------------------------------------------
    # 8. entity resolution
    # ------------------------------------------------------------------------------------

    @step("entities")
    def resolve_counterparty_entities(self) -> str:
        if not self.transactions or len(self.classified) != len(self.transactions):
            return "ERROR: classification must run first."

        raw_names = [item.counterparty for item in self.classified.values() if item.counterparty]

        mapping = canonicalize_names(raw_names)

        changed = 0

        for item in self.classified.values():
            if not item.counterparty:
                continue

            original = item.counterparty
            canonical = mapping.get(original, original)

            if original != canonical:
                changed += 1
                item.counterparty = canonical

        return f"OK resolved={len(mapping)} " f"merged_variants={changed}"

    # ------------------------------------------------------------------------------------
    # 9. relationships
    # ------------------------------------------------------------------------------------

    @step("relationships")
    def detect_transaction_relationships(self) -> str:
        if not self.transactions:
            return "ERROR: no transactions."

        relationships = Counter()

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]

            relationship = item.relationship

            if relationship not in RELATIONSHIPS:
                relationship = self._infer_relationship(
                    tx,
                    item,
                )

            item.relationship = relationship
            relationships[relationship] += 1

        return f"OK relationships={dict(relationships)}"

    # ------------------------------------------------------------------------------------
    # 10. internal transfers
    # ------------------------------------------------------------------------------------

    def _is_possible_internal_transfer(
        self,
        tx: Transaction,
        item: ClassifiedTransaction,
    ) -> bool:
        text = tx.description.upper()

        own_account_terms = (
            "OWN ACCOUNT",
            "SELF TRANSFER",
            "SELF TRF",
            "TRANSFER TO SELF",
            "TRANSFER FROM SELF",
            "OWN A/C",
            "SELF A/C",
        )

        if any(term in text for term in own_account_terms):
            return True

        return item.category == "INTERNAL_TRANSFER"

    # ------------------------------------------------------------------------------------
    # 11. duplicate detection
    # ------------------------------------------------------------------------------------

    @step("duplicates")
    def detect_duplicates(self) -> str:
        if not self.transactions:
            return "ERROR: no transactions."

        groups: dict[tuple, list[Transaction]] = defaultdict(list)

        for tx in self.transactions:
            item = self.classified.get(tx.transaction_id)

            key = (
                tx.date,
                round(tx.amount, 2),
                tx.direction,
                item.counterparty if item else None,
                normalize_name(tx.description),
                tx.reference_number,
            )

            groups[key].append(tx)

        duplicate_groups = 0
        duplicate_rows = 0

        for _, rows in groups.items():
            if len(rows) <= 1:
                continue

            duplicate_groups += 1
            duplicate_rows += len(rows)

            for tx in rows:
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="possible_duplicate",
                        reason=(
                            "Another transaction has the same date, amount, "
                            "direction, counterparty/narration and reference."
                        ),
                        severity="MEDIUM",
                        recommended_review=(
                            "Compare the original statement rows before " "treating this as a duplicate."
                        ),
                    )
                )

        return f"OK duplicate_groups={duplicate_groups} " f"duplicate_rows={duplicate_rows}"

    # ------------------------------------------------------------------------------------
    # 12. cluster + aggregate
    # ------------------------------------------------------------------------------------

    @staticmethod
    def _group_value(
        field: str,
        tx: Transaction,
        item: ClassifiedTransaction,
    ) -> str:
        if field == "direction":
            return tx.direction

        if field == "category":
            return item.category or "UNKNOWN"

        if field == "counterparty":
            return item.counterparty or "UNKNOWN"

        if field == "month":
            return month_of(tx.date) or "UNKNOWN"

        if field == "payment_mode":
            return item.payment_mode or "UNKNOWN"

        if field == "relationship":
            return item.relationship or "UNKNOWN"

        return "UNKNOWN"

    @step("cluster")
    def cluster_and_aggregate(self) -> str:
        if not self.transactions or len(self.classified) != len(self.transactions):
            return "ERROR: transactions are not fully classified. " "Run classify_transactions first."

        account = self.account or MainAccount()

        groups: dict[tuple, list[Transaction]] = defaultdict(list)

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]
            key = tuple(self._group_value(field, tx, item) for field in self.group_by)
            groups[key].append(tx)

        self.clusters = []

        for index, (key, txs) in enumerate(groups.items(), start=1):
            values = dict(zip(self.group_by, key))

            dates = sorted(tx.date for tx in txs if tx.date and ISO_RE.match(tx.date))

            credits = round(
                sum(tx.amount for tx in txs if tx.direction == "CREDIT"),
                2,
            )

            debits = round(
                sum(tx.amount for tx in txs if tx.direction == "DEBIT"),
                2,
            )

            self.clusters.append(
                TransactionCluster(
                    cluster_id=f"CL{index:04d}",
                    account_name=account.account_holder_name,
                    account_number=(account.account_number or account.masked_account),
                    account_type=account.account_type,
                    cluster_type=("COUNTERPARTY" if "counterparty" in self.group_by else "ACCOUNTING"),
                    cluster_name=(
                        values.get("counterparty")
                        or values.get("category")
                        or values.get("relationship")
                        or values.get("month")
                    ),
                    transaction_type=values["direction"],
                    category=values.get("category"),
                    counterparty=values.get("counterparty"),
                    month=values.get("month"),
                    payment_mode=values.get("payment_mode"),
                    relationship=values.get("relationship"),
                    transaction_count=len(txs),
                    credit_total=credits,
                    debit_total=debits,
                    total_amount=round(
                        credits + debits,
                        2,
                    ),
                    first_transaction_date=dates[0] if dates else None,
                    last_transaction_date=dates[-1] if dates else None,
                    recurring=self._is_recurring(txs),
                    confidence=self._cluster_confidence(txs),
                    transaction_ids=[tx.transaction_id for tx in txs],
                )
            )

        self.clusters.sort(
            key=lambda cluster: (
                cluster.transaction_type,
                -cluster.total_amount,
            )
        )

        self._aggregate_counterparties()
        self._aggregate_categories()
        self._aggregate_payment_modes()
        self._aggregate_months()

        return (
            f"OK clusters={len(self.clusters)} "
            f"credits={self.validation.credit_total:,.2f} "
            f"debits={self.validation.debit_total:,.2f}"
        )

    def _is_recurring(self, txs: list[Transaction]) -> bool:
        if len(txs) < 3:
            return False

        dates = sorted(datetime.strptime(tx.date, "%Y-%m-%d") for tx in txs if tx.date and ISO_RE.match(tx.date))

        if len(dates) < 3:
            return False

        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]

        monthly = sum(20 <= gap <= 40 for gap in gaps)

        return monthly >= max(2, len(gaps) // 2)

    def _cluster_confidence(self, txs: list[Transaction]) -> float:
        values = [self.classified[tx.transaction_id].confidence for tx in txs]

        return round(
            sum(values) / len(values) if values else 0.0,
            2,
        )

    def _aggregate_counterparties(self):
        groups: dict[str, list[Transaction]] = defaultdict(list)

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]
            name = item.counterparty or "UNKNOWN"
            groups[name].append(tx)

        summaries = []

        for name, txs in groups.items():
            dates = sorted(tx.date for tx in txs if tx.date and ISO_RE.match(tx.date))

            received = round(
                sum(tx.amount for tx in txs if tx.direction == "CREDIT"),
                2,
            )

            paid = round(
                sum(tx.amount for tx in txs if tx.direction == "DEBIT"),
                2,
            )

            confidence = round(
                sum(self.classified[tx.transaction_id].confidence for tx in txs) / len(txs),
                2,
            )

            cluster_id = next(
                (cluster.cluster_id for cluster in self.clusters if cluster.counterparty == name),
                None,
            )

            summaries.append(
                CounterpartySummary(
                    counterparty_id=(None if name == "UNKNOWN" else f"CP_{abs(hash(name)) % 100000:05d}"),
                    counterparty=name,
                    transaction_count=len(txs),
                    total_received=received,
                    total_paid=paid,
                    net_amount=round(
                        received - paid,
                        2,
                    ),
                    first_transaction=dates[0] if dates else None,
                    last_transaction=dates[-1] if dates else None,
                    payment_modes=sorted({self.classified[tx.transaction_id].payment_mode for tx in txs}),
                    cluster_id=cluster_id,
                    confidence=confidence,
                )
            )

        summaries.sort(
            key=lambda item: (item.total_received + item.total_paid),
            reverse=True,
        )

        self.money_received = [item for item in summaries if item.total_received > 0]

        self.money_paid = [item for item in summaries if item.total_paid > 0]

    def _aggregate_categories(self):
        summary: dict[str, dict[str, float | int]] = {}

        for tx in self.transactions:
            category = self.classified[tx.transaction_id].category

            if category not in summary:
                summary[category] = {
                    "transaction_count": 0,
                    "total_credits": 0.0,
                    "total_debits": 0.0,
                }

            summary[category]["transaction_count"] += 1

            if tx.direction == "CREDIT":
                summary[category]["total_credits"] += tx.amount
            else:
                summary[category]["total_debits"] += tx.amount

        for value in summary.values():
            value["total_credits"] = round(
                float(value["total_credits"]),
                2,
            )
            value["total_debits"] = round(
                float(value["total_debits"]),
                2,
            )

        self.accounting_summary = summary

    def _aggregate_payment_modes(self):
        summary: dict[str, dict[str, float | int]] = {}

        for tx in self.transactions:
            mode = self.classified[tx.transaction_id].payment_mode

            if mode not in summary:
                summary[mode] = {
                    "transaction_count": 0,
                    "total_credits": 0.0,
                    "total_debits": 0.0,
                }

            summary[mode]["transaction_count"] += 1

            if tx.direction == "CREDIT":
                summary[mode]["total_credits"] += tx.amount
            else:
                summary[mode]["total_debits"] += tx.amount

        for value in summary.values():
            value["total_credits"] = round(
                float(value["total_credits"]),
                2,
            )
            value["total_debits"] = round(
                float(value["total_debits"]),
                2,
            )

        self.payment_mode_summary = summary

    def _aggregate_months(self):
        summary: dict[str, dict[str, float | int]] = {}

        for tx in self.transactions:
            month = month_of(tx.date) or "UNKNOWN"

            if month not in summary:
                summary[month] = {
                    "transaction_count": 0,
                    "credits": 0.0,
                    "debits": 0.0,
                    "net_movement": 0.0,
                }

            summary[month]["transaction_count"] += 1

            if tx.direction == "CREDIT":
                summary[month]["credits"] += tx.amount
            else:
                summary[month]["debits"] += tx.amount

        for value in summary.values():
            value["credits"] = round(float(value["credits"]), 2)
            value["debits"] = round(float(value["debits"]), 2)
            value["net_movement"] = round(
                float(value["credits"]) - float(value["debits"]),
                2,
            )

        self.monthly_summary = summary

    # ------------------------------------------------------------------------------------
    # 13. anomalies
    # ------------------------------------------------------------------------------------

    @step("anomalies")
    def detect_anomalies(self) -> str:
        if not self.transactions:
            return "ERROR: no transactions."

        amounts = sorted(tx.amount for tx in self.transactions if tx.amount is not None)

        if not amounts:
            return "ERROR: no transaction amounts."

        median = amounts[len(amounts) // 2]

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]

            if item.confidence < REVIEW_LOW_CONFIDENCE:
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="low_confidence",
                        reason=(f"Classification confidence is " f"{item.confidence:.2f}."),
                        severity="MEDIUM",
                        recommended_review=("Confirm counterparty and accounting purpose."),
                    )
                )

            if not item.counterparty:
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="counterparty_unknown",
                        reason="No reliable payer/payee was identified.",
                        severity="MEDIUM",
                        recommended_review=("Review the original narration/reference."),
                    )
                )

            if item.category == "UNKNOWN":
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="classification_unknown",
                        reason="Accounting purpose could not be determined reliably.",
                        severity="MEDIUM",
                        recommended_review=(
                            "Review the transaction against invoices, bills, "
                            "bank references or supporting documents."
                        ),
                    )
                )

            if median > 0 and tx.amount >= median * 10:
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="unusual_amount",
                        reason=(
                            f"Transaction amount {tx.amount:,.2f} is at least "
                            f"10x the statement transaction median "
                            f"{median:,.2f}."
                        ),
                        severity="MEDIUM",
                        recommended_review=("Check supporting documentation and business context."),
                    )
                )

            if self._is_possible_internal_transfer(tx, item):
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="possible_internal_transfer",
                        reason="Narration/category suggests a transfer between own accounts.",
                        severity="LOW",
                        recommended_review=(
                            "Confirm the destination/source account belongs " "to the same account holder."
                        ),
                    )
                )

            if item.category == "PERSONAL_EXPENSE" and tx.direction == "DEBIT":
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type="possible_personal_expense",
                        reason="The transaction was classified as a possible personal expense.",
                        severity="MEDIUM",
                        recommended_review=("Confirm whether the payment is business-related."),
                    )
                )

        self.anomalies = list(self.review_queue)

        return f"OK review_candidates={len(self.review_queue)} " f"median_amount={median:,.2f}"

    # ------------------------------------------------------------------------------------
    # 14. CA review queue
    # ------------------------------------------------------------------------------------

    @step("review")
    def generate_ca_review_queue(self) -> str:
        if not self.transactions:
            return "ERROR: no transactions."

        # Validation issues are statement-level; attach them to the first
        # relevant transaction only when a specific row cannot be identified.
        if self.validation.balance_consistency_percentage < 90:
            self._add_review(
                ReviewItem(
                    transaction_id=None,
                    issue_type="balance_mismatch",
                    reason=(
                        "Running balance consistency is " f"{self.validation.balance_consistency_percentage:.1f}%."
                    ),
                    severity="HIGH" if self.validation.balance_consistency_percentage < 70 else "MEDIUM",
                    recommended_review=(
                        "Recheck extraction, transaction ordering, " "debit/credit columns and printed balances."
                    ),
                )
            )

        for tx in self.transactions:
            item = self.classified[tx.transaction_id]

            if not tx.reference_number:
                extracted = extract_reference(tx.description)
                if not extracted:
                    self._add_review(
                        ReviewItem(
                            transaction_id=tx.transaction_id,
                            issue_type="missing_reference",
                            reason="No transaction reference/UTR was extracted.",
                            severity="LOW",
                            recommended_review=("Check whether the source statement contains " "a reference number."),
                        )
                    )

            if item.relationship in {"REFUND", "REVERSAL"}:
                self._add_review(
                    ReviewItem(
                        transaction_id=tx.transaction_id,
                        issue_type=("possible_refund" if item.relationship == "REFUND" else "possible_reversal"),
                        reason=(f"Transaction relationship is " f"{item.relationship}."),
                        severity="MEDIUM",
                        recommended_review=(
                            "Locate the original transaction and confirm " "the reversal/refund relationship."
                        ),
                    )
                )

        self.review_queue = self._deduplicate_review_items()

        return f"OK review_items={len(self.review_queue)}"

    def _add_review(self, item: ReviewItem):
        self.review_queue.append(item)

    def _deduplicate_review_items(self) -> list[ReviewItem]:
        seen = set()
        output = []

        for item in self.review_queue:
            key = (
                item.transaction_id,
                item.issue_type,
                item.reason,
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(item)

        return output

    # ------------------------------------------------------------------------------------
    # 15. final summary
    # ------------------------------------------------------------------------------------

    @step("summary")
    def generate_final_summary(self) -> str:
        if not self.transactions:
            return "ERROR: no transactions."

        account = self.account or MainAccount()

        unknown_counterparties = sum(1 for item in self.classified.values() if not item.counterparty)

        unknown_categories = sum(1 for item in self.classified.values() if item.category == "UNKNOWN")

        summary_lines = [
            (
                f"Processed {len(self.transactions)} transactions "
                f"for {account.account_holder_name or 'the account'}."
            ),
            (
                f"Credits={self.validation.credit_total:,.2f}; "
                f"Debits={self.validation.debit_total:,.2f}; "
                f"Net movement="
                f"{self.validation.credit_total - self.validation.debit_total:,.2f}."
            ),
            (
                f"Identified {len(self.money_received)} incoming counterparty groups "
                f"and {len(self.money_paid)} outgoing counterparty groups."
            ),
            (
                f"{unknown_counterparties} transactions have no reliable counterparty "
                f"and {unknown_categories} remain unclassified."
            ),
            (
                f"CA review queue contains {len(self.review_queue)} items; "
                f"validation status={self.validation.validation_status}."
            ),
        ]

        self.final_summary = "\n".join(summary_lines)
        return self.final_summary

    # ------------------------------------------------------------------------------------
    # Final object
    # ------------------------------------------------------------------------------------

    def build_result(self) -> StatementResult:
        account = self.account or MainAccount()

        dates = sorted(tx.date for tx in self.transactions if tx.date and ISO_RE.match(tx.date))

        if not account.statement_period and dates:
            account.statement_period = f"{dates[0]} to {dates[-1]}"

        if account.statement_period_start is None and dates:
            account.statement_period_start = dates[0]

        if account.statement_period_end is None and dates:
            account.statement_period_end = dates[-1]

        total_credit = round(
            sum(tx.amount for tx in self.transactions if tx.direction == "CREDIT"),
            2,
        )

        total_debit = round(
            sum(tx.amount for tx in self.transactions if tx.direction == "DEBIT"),
            2,
        )

        warnings = [warning for warnings_for_step in self.warn.values() for warning in warnings_for_step]

        flat_transactions = None

        if self.include_transactions:
            flat_transactions = []

            cluster_lookup = {}

            for cluster in self.clusters:
                for transaction_id in cluster.transaction_ids:
                    cluster_lookup[transaction_id] = cluster

            for tx in self.transactions:
                item = self.classified[tx.transaction_id]
                cluster = cluster_lookup.get(tx.transaction_id)

                description_normalized = normalize_name(tx.description)

                row = {
                    # RAW
                    "transaction_id": tx.transaction_id,
                    "account_id": (account.account_number or account.masked_account),
                    "transaction_date": tx.date,
                    "value_date": tx.value_date,
                    "date_original": tx.date_original,
                    "description_raw": tx.description,
                    "amount": tx.amount,
                    "direction": tx.direction,
                    "balance": tx.balance,
                    "currency": tx.currency or account.currency,
                    "reference_number": tx.reference_number,
                    "cheque_number": tx.cheque_number,
                    "source": {
                        "page": tx.source_page,
                        "row": tx.source_row,
                        "sheet": tx.source_sheet,
                    },
                    # NORMALIZED
                    "description_normalized": description_normalized,
                    "payment_mode": item.payment_mode,
                    "utr_number": extract_reference(tx.description),
                    "upi_id": extract_upi(tx.description),
                    "merchant_name_raw": (
                        item.counterparty if item.counterparty_type in {"BUSINESS", "PLATFORM"} else None
                    ),
                    "counterparty_name_raw": item.counterparty,
                    # INTELLIGENCE
                    "counterparty": {
                        "counterparty_id": (
                            None if not item.counterparty else f"CP_{abs(hash(item.counterparty)) % 100000:05d}"
                        ),
                        "name_raw": item.counterparty,
                        "name_normalized": item.counterparty,
                        "type": item.counterparty_type,
                        "confidence": item.confidence,
                    },
                    "relationship": item.relationship,
                    "cluster": (
                        {
                            "cluster_id": cluster.cluster_id,
                            "cluster_type": cluster.cluster_type,
                            "cluster_name": cluster.cluster_name,
                            "confidence": cluster.confidence,
                        }
                        if cluster
                        else None
                    ),
                    "classification": {
                        "category": item.category,
                        "subcategory": item.subcategory,
                        "confidence": item.confidence,
                        "reason": item.classification_reason,
                    },
                    "income_expense_interpretation": self._income_expense_interpretation(
                        tx,
                        item,
                    ),
                    "flags": {
                        "possible_duplicate": any(
                            review.transaction_id == tx.transaction_id and review.issue_type == "possible_duplicate"
                            for review in self.review_queue
                        ),
                        "possible_internal_transfer": self._is_possible_internal_transfer(
                            tx,
                            item,
                        ),
                        "possible_personal": (item.category == "PERSONAL_EXPENSE"),
                        "requires_review": any(
                            review.transaction_id == tx.transaction_id for review in self.review_queue
                        ),
                    },
                    "extraction_confidence": tx.extraction_confidence,
                }

                flat_transactions.append(row)

        data_quality = {
            "extraction_confidence": round(
                (
                    sum(tx.extraction_confidence or 0 for tx in self.transactions) / len(self.transactions)
                    if self.transactions
                    else 0
                ),
                2,
            ),
            "balance_consistency": self.validation.balance_consistency_percentage,
            "date_consistency": self.validation.date_consistency_percentage,
            "missing_information_count": self.validation.missing_data_count,
            "duplicate_candidates": self.validation.duplicate_row_count,
            "ocr_issues": [],
            "ambiguous_transactions": sum(
                1 for item in self.classified.values() if item.confidence < REVIEW_LOW_CONFIDENCE
            ),
            "unclassified_transactions": sum(1 for item in self.classified.values() if item.category == "UNKNOWN"),
        }

        self.result = StatementResult(
            account=account,
            group_by=self.group_by,
            transaction_count=len(self.transactions),
            total_credit=total_credit,
            total_debit=total_debit,
            net_movement=round(
                total_credit - total_debit,
                2,
            ),
            warnings=warnings,
            validation=self.validation,
            clusters=self.clusters,
            money_received=self.money_received,
            money_paid=self.money_paid,
            accounting_summary=self.accounting_summary,
            payment_mode_summary=self.payment_mode_summary,
            monthly_summary=self.monthly_summary,
            review_queue=self.review_queue,
            anomalies=self.anomalies,
            data_quality=data_quality,
            transactions=flat_transactions,
            summary=self.final_summary,
        )

        return self.result

    def _income_expense_interpretation(
        self,
        tx: Transaction,
        item: ClassifiedTransaction,
    ) -> str:
        if item.category == "INTERNAL_TRANSFER":
            return "internal_transfer"

        if item.category in {"LOAN_RECEIPT"}:
            return "loan"

        if item.category in {"LOAN_REPAYMENT"}:
            return "loan"

        if item.category in {
            "INVESTMENT",
            "DIVIDEND",
            "INTEREST_INCOME",
        }:
            return "investment"

        if item.category in {
            "GST_PAYMENT",
            "TDS_PAYMENT",
            "TAX_PAYMENT",
            "GOVERNMENT_PAYMENT",
        }:
            return "tax"

        if item.category == "REFUND":
            return "refund"

        if item.category == "PERSONAL_EXPENSE":
            return "personal"

        if tx.direction == "CREDIT":
            if item.category == "SALES_RECEIPT":
                return "business_income"
            return "unknown"

        if tx.direction == "DEBIT":
            if item.category not in {
                "UNKNOWN",
                "INTERNAL_TRANSFER",
                "LOAN_REPAYMENT",
            }:
                return "business_expense"

        return "unknown"

    # ------------------------------------------------------------------------------------
    # Fixed pipeline
    # ------------------------------------------------------------------------------------

    def run_all(self):
        steps = [
            ("inspect", self.inspect_statement),
            ("boundaries", self.detect_boundaries),
            ("account", self.extract_account),
            ("layout", self.discover_layout),
            ("transactions", self.extract_transactions),
            ("validate", self.validate_transactions),
            ("classify", self.classify_transactions),
            ("counterparties", self.identify_counterparties),
            ("entities", self.resolve_counterparty_entities),
            ("relationships", self.detect_transaction_relationships),
            ("duplicates", self.detect_duplicates),
            ("cluster", self.cluster_and_aggregate),
            ("anomalies", self.detect_anomalies),
            ("review", self.generate_ca_review_queue),
            ("summary", self.generate_final_summary),
        ]

        # Account extraction and validation are allowed to finish with warnings.
        non_fatal = {
            "inspect",
            "account",
            "validate",
            "anomalies",
            "review",
            "summary",
        }

        extraction_retried = False

        for name, function in steps:
            if name in self.done:
                continue

            result = function()

            if result.startswith("ERROR"):
                # The old code already used smaller extraction chunks as the
                # recovery path. Keep that behavior, but do it deterministically.
                if name == "transactions" and not extraction_retried and self.chunk_size > 1800:
                    extraction_retried = True
                    retry_result = self.extract_transactions(max(1800, self.chunk_size // 2))
                    if retry_result.startswith("ERROR"):
                        raise RuntimeError(retry_result)
                    continue

                if name == "validate":
                    # Validation can warn without stopping intelligence.
                    continue

                if name in non_fatal:
                    continue

                raise RuntimeError(result)

        # Material validation failure should not silently become intelligence.
        if self.validation.validation_status == "FAIL" and len(self.transactions) > 0:
            self.warn.setdefault("pipeline", []).append(
                "Transaction extraction validation failed materially. "
                "Review extraction before relying on accounting conclusions."
            )

        return self.build_result()


# ----------------------------------------------------------------------------------------
# Compatibility / external API
# ----------------------------------------------------------------------------------------


def build_tools(p: StatementPipeline):
    """
    Kept for compatibility with the old code.

    The normal implementation does not need LangGraph. These are plain callable
    wrappers so an existing caller can still expose the pipeline as tools.
    """
    return [
        p.inspect_statement,
        p.detect_boundaries,
        p.extract_account,
        p.discover_layout,
        p.extract_transactions,
        p.validate_transactions,
        p.classify_transactions,
        p.identify_counterparties,
        p.resolve_counterparty_entities,
        p.detect_transaction_relationships,
        p.detect_duplicates,
        p.cluster_and_aggregate,
        p.detect_anomalies,
        p.generate_ca_review_queue,
        p.generate_final_summary,
    ]


def run_agent(p: StatementPipeline):
    """
    Compatibility function for callers of the old implementation.

    There is intentionally no ReAct/LangGraph loop here. The ordered pipeline
    is more reliable for this deterministic ETL-style workload.
    """
    return p.run_all()


def process_statement(
    data: str,
    group_by: Optional[list[str]] = None,
    include_transactions: bool = False,
    use_agent: bool = False,
) -> StatementResult:
    """
    Main entry point.

    use_agent is retained for API compatibility. The current implementation
    deliberately uses the fixed pipeline because every stage has a known
    dependency and should execute in order.
    """
    llm = get_ollama_instance()

    pipeline = StatementPipeline(
        data,
        llm,
        group_by=group_by,
        include_transactions=include_transactions,
    )

    return pipeline.run_all()
