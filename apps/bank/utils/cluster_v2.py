"""
Bank Statement Intelligence Engine
==================================

Turns the text of a bank statement into auditable transaction intelligence:
who paid us, whom we paid, which transactions belong together, what the money
was probably for, and what a Chartered Accountant (CA) still has to verify.

This module implements two design documents, referred to below as:

    LOGIC  = bank_statement_grouping_logic_and_prompts   (sections written §n)
    GUIDE  = bank_statement_transaction_grouping_guide   (sections written §n)

--------------------------------------------------------------------------
THE ONE RULE  (LOGIC §20)
--------------------------------------------------------------------------
Keep these six questions separate. Each has its own stage and its own fields:

    Extraction         What did the bank print?
    Normalization      What structured fields can we derive?
    Entity resolution  Who is involved?                       (counterparty_id)
    Clustering         Which transactions appear related?     (cluster_id)
    Classification     What was the money probably for?       (category)
    CA review          What still needs human verification?   (review_queue)

--------------------------------------------------------------------------
PIPELINE  (LOGIC §1)
--------------------------------------------------------------------------
    extract_account        header facts (bank, holder, balances)
    extract_transactions   Extraction            LLM   -> raw transactions
    validate               Validation            math  (LOGIC §1, §2)
    normalize_and_resolve  Normalization         regex + LLM (narration reading)
                           Relationship det.     LLM   (transfer/refund/wallet...)
                           Entity resolution     code  (LOGIC §3, §4, §5)
                           Internal transfers    code  (LOGIC §15)
    detect_duplicates      Duplicate detection   code  (LOGIC §14)
    build_clusters         Clustering            code  (LOGIC §6, GUIDE §4, §5)
    classify               Classification        LLM   (LOGIC §11, GUIDE §8)
    build_review_queue     CA review             code  (LOGIC §17)
    aggregate              Aggregation           code  (LOGIC §12, §13, §16)

Order note: LOGIC §1 draws "Aggregation -> Classification", but LOGIC §16
asks for totals *by category*, which need the classification first. So
classification and review run before aggregation. Nothing else changes.

--------------------------------------------------------------------------
WHO DOES WHAT  (LOGIC "Implementation note")
--------------------------------------------------------------------------
Deterministic code owns: identifiers (UPI id, IFSC, reference), counterparty
ids, match decisions, clusters, duplicates, totals, aggregation, review flags.
The LLM owns only messy-narration interpretation (names, type, relationship),
and accounting classification, and always returns evidence.

Why clustering has no LLM call: LOGIC §1 says "the LLM should not be the only
mechanism used for clustering; use deterministic identifiers first". The rules
of the clustering prompt (LOGIC §10) are enforced directly in build_clusters:
a cluster only holds transaction ids and is never created from weak name
similarity alone.

--------------------------------------------------------------------------
GUARANTEES
--------------------------------------------------------------------------
* A transaction is never merged, changed or deleted. Clusters, duplicate
  groups and review items only *link* to transaction ids (GUIDE §1, §8).
* The LLM may not invent a counterparty: a name must appear in the narration
  (_grounded).
* Name similarity alone never merges two counterparties. It gives POSSIBLE +
  review (LOGIC §3, §4).
* Counterparty, cluster and classification are separate fields. When evidence
  is weak the category is UNKNOWN and the transaction is flagged (GUIDE §8).
* Every derived decision keeps its evidence; the result carries the model name
  and PROMPT_VERSION (LOGIC "Implementation note").
"""

import logging
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Optional

from pydantic import BaseModel, Field

from apps.common.llms import get_ollama_instance

logger = logging.getLogger(__name__)

PROMPT_VERSION = "3.0.0"

# ============================================================
# CONSTANTS
# ============================================================

# LOGIC §5 - example evidence weights. These are starting points, not
# accounting rules: calibrate them on real statements by measuring false
# merges and missed merges.
# "strong_id" (IFSC / reference match) is NOT in the LOGIC table; it is added
# so a STRONG match scores above a POSSIBLE one. It equals the merchant weight.
WEIGHTS = {
    "account": 0.50,
    "upi": 0.30,
    "merchant": 0.25,
    "strong_id": 0.25,
    "name": 0.15,  # strong normalized-name match
    "narration": 0.10,  # same narration pattern seen before
    "history": 0.10,  # same payment mode + direction seen before
}

# LOGIC §3 evidence order 1-3: identifier prefix -> weight key.
EXACT_KINDS = {"acct:": "account", "upi:": "upi", "mid:": "merchant"}
# LOGIC §4 "strong_identifier_match": shared bank code (IFSC) or reference.
STRONG_KINDS = ("ifsc:", "ref:")

NAME_SIMILARITY = 0.85  # LOGIC §4 "name_similarity_high"
DUPLICATE_NARRATION_SIMILARITY = 0.90  # LOGIC §14 "highly_similar_narration"
BALANCE_TOLERANCE = 0.02  # rupees, for the running-balance check
CLASSIFY_MIN_CONFIDENCE = 0.40  # below this the category becomes UNKNOWN
REVIEW_LOW_CONFIDENCE = 0.60  # below this extraction/classification is reviewed
PERSONAL_MIN_CONFIDENCE = 0.80  # PERSONAL_EXPENSE below this is "evidence weak"

PAYMENT_MODES = [
    "UPI", "NEFT", "RTGS", "IMPS", "CHEQUE", "CASH", "ATM", "POS", "CARD",
    "NACH", "ECS", "INTEREST", "BANK_CHARGE", "UNKNOWN",
]  # fmt: skip

# Relationship detection (LOGIC §2): transfers, refunds, reversals, charges,
# wallet top-ups and similar.
RELATIONSHIPS = [
    "PAYMENT", "RECEIPT", "REFUND", "REVERSAL", "TRANSFER", "LOAN_DISBURSEMENT",
    "LOAN_REPAYMENT", "INTEREST", "CHARGE", "TAX_PAYMENT", "SALARY",
    "WALLET_TOP_UP", "CASH", "UNKNOWN",
]  # fmt: skip

COUNTERPARTY_TYPES = ["PERSON", "BUSINESS", "BANK", "GOVERNMENT", "PLATFORM", "UNKNOWN"]

# LOGIC §11 - exactly the categories listed there.
CATEGORIES = [
    "SALES_RECEIPT", "PURCHASE", "RAW_MATERIAL", "INVENTORY", "SALARY", "RENT",
    "PROFESSIONAL_FEES", "TRAVEL", "UTILITIES", "SOFTWARE_SUBSCRIPTION",
    "ADVERTISING", "BANK_CHARGES", "INTEREST", "LOAN_REPAYMENT", "LOAN_RECEIPT",
    "TAX_PAYMENT", "GST_PAYMENT", "TDS_PAYMENT", "INSURANCE", "REFUND",
    "REIMBURSEMENT", "INTERNAL_TRANSFER", "PERSONAL_EXPENSE",
    "CAPITAL_EXPENDITURE", "INVESTMENT", "INTEREST_INCOME", "CASH_DEPOSIT",
    "CASH_WITHDRAWAL", "UNKNOWN",
]  # fmt: skip

# Deterministic payment-mode detection from the narration (first match wins).
_MODE_PATTERNS = [
    ("UPI", r"\bUPI\b"),
    ("IMPS", r"\bIMPS\b"),
    ("NEFT", r"\bNEFT\b"),
    ("RTGS", r"\bRTGS\b"),
    ("NACH", r"\bNACH\b"),
    ("ECS", r"\bECS\b"),
    ("CHEQUE", r"\b(CHQ|CHEQUE|CLG|CLEARING)\b"),
    ("ATM", r"\bATM\b"),
    ("POS", r"\bPOS\b"),
    ("CARD", r"\b(DEBIT|CREDIT) CARD\b"),
    ("CASH", r"\bCASH\b"),
    ("INTEREST", r"\bINT(EREST)?\b"),
    ("BANK_CHARGE", r"\b(CHARGES?|CHGS?|FEE)\b"),
]
_IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_UPI_RE = re.compile(r"\b[\w.\-]{2,}@[A-Za-z]{2,}\b")
_REF_RE = re.compile(r"\b\d{9,22}\b")  # UPI RRN, IMPS/NEFT reference, UTR


# ============================================================
# SCHEMAS
# ============================================================


class MainAccount(BaseModel):
    """Account-level facts printed on the statement."""

    bank_name: Optional[str] = None
    account_holder_name: Optional[str] = None
    account_number: Optional[str] = None
    ifsc: Optional[str] = None
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None


class RawTransaction(BaseModel):
    """Extraction output: what the bank printed. Never edited afterwards."""

    date: Optional[str] = None  # ISO YYYY-MM-DD
    description: str  # narration exactly as printed
    amount: float  # positive
    direction: str  # DEBIT | CREDIT
    balance: Optional[float] = None
    reference_number: Optional[str] = None
    source_line: Optional[int] = None  # audit trail: line where the row starts
    source_page: Optional[str] = None  # audit trail: statement page
    extraction_confidence: Optional[float] = None


class TransactionExtraction(BaseModel):
    transactions: list[RawTransaction]


class Transaction(RawTransaction):
    transaction_id: str  # e.g. TXN_00045


class NarrationReading(BaseModel):
    """What the LLM reads from one narration (LLM part of normalization).

    Nothing here is an accounting judgment (LOGIC §9: "You are NOT deciding
    the final accounting treatment").
    """

    transaction_id: str
    payment_mode: str = "UNKNOWN"  # only used if the regex finds no mode
    counterparty_name_raw: Optional[str] = None  # as printed; null if absent
    counterparty_alias: Optional[str] = None  # second printed name, e.g. MANI ANNA
    counterparty_account: Optional[str] = None
    merchant_id: Optional[str] = None  # platform / merchant identifier, e.g. PAYTM
    counterparty_type: str = "UNKNOWN"
    relationship: str = "UNKNOWN"
    own_account_transfer_suspected: bool = False
    possible_existing_id: Optional[str] = None  # an existing CP that MIGHT match
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class NarrationBatch(BaseModel):
    items: list[NarrationReading]


class Normalized(NarrationReading):
    """Normalized information for one transaction (GUIDE §2), raw data untouched.

    Adds the deterministic fields (regex) to the LLM reading. `reference_number`
    also holds the UTR (LOGIC lists both; they are the same kind of value).
    """

    reference_number: Optional[str] = None
    bank_ifsc: Optional[str] = None
    upi_id: Optional[str] = None
    normalized_name: str = ""  # lowercase alphanumerics only


class Resolution(BaseModel):
    """Entity-resolution result for one transaction (LOGIC §9 output)."""

    transaction_id: str
    counterparty_id: Optional[str] = None
    counterparty_name: Optional[str] = None  # canonical name of the counterparty
    counterparty_name_raw: Optional[str] = None  # as printed in this narration
    counterparty_name_normalized: Optional[str] = None
    counterparty_type: str = "UNKNOWN"
    match_type: str = "UNKNOWN"  # EXACT | STRONG | POSSIBLE | NEW | UNKNOWN
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    possible_match_of: Optional[str] = None  # counterparty id, never merged
    requires_review: bool = False


class Classification(BaseModel):
    """Accounting classification (LOGIC §11 output)."""

    transaction_id: str
    category: str = "UNKNOWN"
    subcategory: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    requires_review: bool = False


class ClassificationBatch(BaseModel):
    items: list[Classification]


class EntityCluster(BaseModel):
    """Parent analytical object linking transactions (LOGIC §6). Holds ids only."""

    cluster_id: str
    cluster_type: str  # COUNTERPARTY|MERCHANT|RECURRING_PAYMENT|WALLET|TAX
    cluster_name: str
    counterparty_id: str
    transaction_ids: list[str]
    transaction_count: int
    total_credit: float
    total_debit: float
    net_amount: float
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    recurring: bool = False
    cluster_confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    possible_related_cluster_ids: list[str] = Field(default_factory=list)
    requires_review: bool = False


class Counterparty(BaseModel):
    """One party, with LOGIC §16 per-counterparty aggregation."""

    counterparty_id: str
    name: str
    counterparty_type: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    transaction_count: int = 0
    total_credit: float = 0.0
    total_debit: float = 0.0
    net_amount: float = 0.0
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None


class CounterpartySummary(BaseModel):
    """One row of 'Who paid us?' (LOGIC §12) or 'Whom did we pay?' (LOGIC §13)."""

    counterparty_id: Optional[str] = None
    counterparty: str
    cluster_id: Optional[str] = None
    transaction_count: int
    total: float
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    payment_modes: list[str] = Field(default_factory=list)
    transaction_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    review_flags: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    transaction_id: Optional[str] = None
    issue_type: str
    reason: str
    severity: str = "MEDIUM"  # HIGH | MEDIUM | LOW


class ValidationResult(BaseModel):
    transaction_count: int = 0
    credit_total: float = 0.0
    debit_total: float = 0.0
    balance_consistency_percentage: float = 100.0
    directions_corrected: int = 0
    balance_break_transaction_ids: list[str] = Field(default_factory=list)
    validation_status: str = "WARNING"  # PASS | WARNING | FAIL
    issues: list[str] = Field(default_factory=list)


class StatementResult(BaseModel):
    """Follows the CA dashboard order (LOGIC §19):

    Account Summary -> Money Received -> Money Paid -> Counterparties ->
    Clusters -> Categories -> Monthly Trends -> Unknown Transactions ->
    Review Queue.
    """

    account: MainAccount
    transaction_count: int
    total_credit: float
    total_debit: float
    net_movement: float
    validation: ValidationResult
    money_received: list[CounterpartySummary] = Field(default_factory=list)
    money_paid: list[CounterpartySummary] = Field(default_factory=list)
    category_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    counterparties: list[Counterparty] = Field(default_factory=list)
    clusters: list[EntityCluster] = Field(default_factory=list)
    categories: dict = Field(default_factory=dict)
    payment_modes: dict = Field(default_factory=dict)
    monthly_trends: dict = Field(default_factory=dict)
    unknown_transaction_ids: list[str] = Field(default_factory=list)
    review_queue: list[ReviewItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    # Per-transaction tree (LOGIC §18): raw / normalized / counterparty /
    # cluster / classification / review_flags. Optional because it is large.
    transactions: Optional[list[dict[str, Any]]] = None


# ============================================================
# PROMPTS
# ============================================================

ACCOUNT_PROMPT = """Extract account-level metadata from this bank statement text (header first, then the end of the statement).
Use null for anything not printed. Dates as YYYY-MM-DD. Balances as plain numbers. Never invent values.

{text}"""

EXTRACT_PROMPT = """You are the Transaction Extraction Agent for a bank statement. The lines below are numbered (L<n>).

Extract ONLY the transactions whose first line number is between L{start} and L{end} inclusive.
Lines outside that range are context (a row may continue onto the following lines, or a previous row's
tail may appear first: do not extract those).

RULES
- One object per transaction. Never merge, skip or invent rows.
- description: the narration exactly as printed (join wrapped lines with a single space).
- amount: positive number. direction: DEBIT or CREDIT. Decide from the table layout, Dr/Cr markers, or the change in running balance.
- date: YYYY-MM-DD. balance: running balance after the row if printed. reference_number only if printed.
- source_line: the line number where this row starts (number only). source_page: from a '=== PAGE n ===' marker if present.
- Ignore headers, footers, column titles, opening/closing balance lines, and totals.

{text}"""

# LOGIC §9 (Counterparty Resolution master prompt). The LLM only READS the
# narration. Counterparty ids and match types are decided by code
# (_resolve), so those output fields are not requested here.
ENRICH_PROMPT = """You are the Counterparty Resolution Engine for a bank-statement processing system.

Your task is to identify the most likely party associated with each bank transaction.
You are NOT deciding the final accounting treatment.

For CREDIT transactions, determine who appears to have paid the account holder.
For DEBIT transactions, determine whom the account holder appears to have paid.
Use only evidence present in the transaction and supplied historical counterparty records.

Account holder: {holder} (account {account})

EVIDENCE PRIORITY:
1. Exact bank account number
2. Exact UPI ID
3. Exact merchant/platform identifier
4. Exact reliable reference/UTR identifier
5. Normalized counterparty name
6. Narration similarity
7. Historical transaction pattern

RULES:
- Never invent a counterparty. If the narration has no party, leave counterparty_name_raw null.
- Never merge parties solely because names are similar.
- Preserve raw narration.
- If evidence is insufficient, return UNKNOWN.
- If a possible match exists but is not sufficiently reliable, set possible_existing_id to that id.
  A human reviews it; nothing is merged automatically.
- Do not infer personal/business status solely from the name. counterparty_type is UNKNOWN unless clear.
- own_account_transfer_suspected is true ONLY with evidence such as the holder's own name/account or "self".
  Do not call a transfer internal merely because the amount is similar.
- A wallet load such as "ADD MONEY" has relationship WALLET_TOP_UP; it is not a purchase.

For each transaction return:
- payment_mode: one of {modes}
- counterparty_name_raw: the party's name AS PRINTED; counterparty_alias if a second name is printed
- counterparty_account: payer/payee account number, only if present
- merchant_id: platform/merchant/tax-authority identifier such as PAYTM or GST, only if present
- counterparty_type: one of {types}
- relationship: one of {relationships}
- own_account_transfer_suspected, possible_existing_id
- confidence 0-1 and evidence: short quotes from the narration

Existing counterparties: {existing}

Transactions:
{items}"""

# LOGIC §11 (Accounting Classification master prompt) + GUIDE §8 rule.
CLASSIFY_PROMPT = """You are the Accounting Classification Engine.

Determine the most likely business/accounting purpose of a bank transaction AFTER counterparty
resolution and clustering.

Do not treat CREDIT as income automatically.
Do not treat DEBIT as expense automatically.
Do not classify solely from the cluster name.

Possible classifications: {categories}

Rules:
- Use UNKNOWN when evidence is insufficient.
- Never invent an invoice, GST number, business purpose or expense purpose.
- Keep classification separate from counterparty and cluster. Cluster context is a hint, not proof.
- A wallet top-up (ADD MONEY) is not an expense: it may only load a wallet.
- Flag ambiguous transactions for CA review (requires_review=true).
- evidence: short quotes or facts you relied on.

Transactions:
{items}"""


# ============================================================
# HELPERS
# ============================================================


def _iso(v) -> Optional[str]:
    try:
        return datetime.fromisoformat(str(v)[:10]).date().isoformat()
    except (ValueError, TypeError):
        return None


def _month(v) -> Optional[str]:
    d = _iso(v)
    return d[:7] if d else None


def _enum(v, allowed: list[str]) -> str:
    """Coerce an LLM value into an allowed value, else UNKNOWN."""
    s = str(v or "").strip().upper().replace(" ", "_").replace("-", "_")
    return s if s in allowed else "UNKNOWN"


def _key(name: Optional[str]) -> str:
    """Normalized name: lowercase alphanumeric tokens joined by one space."""
    return " ".join("".join(c if c.isalnum() else " " for c in (name or "").lower()).split())


def _grounded(name: Optional[str], narration: str) -> bool:
    """Anti-hallucination ("never invent a counterparty"): at least one
    meaningful token of the name must appear in the narration."""
    tokens = [t for t in _key(name).split() if len(t) >= 3]
    return bool(tokens) and any(t in narration.lower() for t in tokens)


def _narration_pattern(narration: str) -> str:
    """Narration with digits removed, so repeated payments to the same party
    produce the same pattern (LOGIC §3 "repeated normalized narration")."""
    return _key(re.sub(r"\d+", " ", narration))


def _names_similar(key: str, known: set[str]) -> bool:
    """LOGIC §4 name_similarity_high: equal, one name's meaningful tokens
    contained in the other's (MANIKANDAN vs MANIKANDAN R), or text ratio
    >= NAME_SIMILARITY. This is evidence, never proof."""
    if not key:
        return False
    sig = {t for t in key.split() if len(t) >= 3}
    for other in known:
        if not other:
            continue
        osig = {t for t in other.split() if len(t) >= 3}
        if key == other:
            return True
        if sig and osig and (sig <= osig or osig <= sig):
            return True
        if SequenceMatcher(None, key, other).ratio() >= NAME_SIMILARITY:
            return True
    return False


def _digits(v: Optional[str]) -> str:
    return re.sub(r"\D", "", v or "")


def _totals(ts: list[Transaction]) -> dict:
    """LOGIC §16: count, sum credits, sum debits, net movement."""
    cr = round(sum(t.amount for t in ts if t.direction == "CREDIT"), 2)
    dr = round(sum(t.amount for t in ts if t.direction == "DEBIT"), 2)
    return {"count": len(ts), "credit": cr, "debit": dr, "net": round(cr - dr, 2)}


def _date_range(ts: list[Transaction]) -> tuple[Optional[str], Optional[str]]:
    dates = sorted(t.date for t in ts if t.date)
    return (dates[0], dates[-1]) if dates else (None, None)


# ============================================================
# PIPELINE
# ============================================================


class StatementPipeline:
    def __init__(self, raw_text: str, llm, include_transactions=False, chunk_chars=3500, batch_size=20):
        # Non-empty lines; the index in this list is the "line number" (L<n>)
        # used for extraction and stored as source_line for the audit trail.
        self.lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
        self.llm = llm
        self.model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__
        self.include_transactions = include_transactions
        self.chunk_chars, self.batch_size = chunk_chars, batch_size

        self.warnings: list[str] = []
        self.account = MainAccount()
        self.tx: list[Transaction] = []
        self.validation = ValidationResult()

        self.norm: dict[str, Normalized] = {}
        self.res: dict[str, Resolution] = {}
        # Counterparty registry: id -> {name, type, ids, names, aliases, modes,
        # directions, patterns, count}. `ids` = identifiers seen for this party.
        self.cps: dict[str, dict] = {}
        self._id_index: dict[str, str] = {}  # exact identifier -> counterparty id

        self.dup: dict[str, dict] = {}
        self.by_cp: dict[str, list[Transaction]] = {}
        self.clusters: list[EntityCluster] = []
        self.cluster_of: dict[str, str] = {}  # transaction id -> cluster id
        self.cls: dict[str, Classification] = {}
        self.review: list[ReviewItem] = []
        self.flags: dict[str, set] = defaultdict(set)  # transaction id -> issue types

        self.counterparties: list[Counterparty] = []
        self.received: list[CounterpartySummary] = []
        self.paid: list[CounterpartySummary] = []
        self.category_breakdown: dict[str, dict[str, float]] = {}
        self.categories: dict = {}
        self.modes: dict = {}
        self.months: dict = {}

    # -- LLM plumbing ---------------------------------------------------------

    def _ask(self, schema, prompt: str, retries: int = 2):
        last = None
        for _ in range(retries + 1):
            try:
                out = self.llm.with_structured_output(schema).invoke(prompt)
                return schema.model_validate(out) if isinstance(out, dict) else out
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"LLM call failed: {last}")

    def _run(self, name: str, fn, fatal: bool = False):
        """Run one stage. A non-fatal failure becomes a warning; the pipeline continues."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.exception("stage %s failed", name)
            if fatal:
                raise RuntimeError(f"{name} failed: {e}") from e
            self.warnings.append(f"Stage '{name}' failed and was skipped: {e}")

    # ========================================================================
    # 1. ACCOUNT
    # ========================================================================

    def extract_account(self):
        text = "\n".join(self.lines)
        self.account = self._ask(MainAccount, ACCOUNT_PROMPT.format(text=f"{text[:6000]}\n...\n{text[-1500:]}"))

    # ========================================================================
    # 2. EXTRACTION  - "What did the bank print?"
    # ========================================================================

    def _ranges(self) -> list[tuple[int, int]]:
        """Split the statement into chunks [start, end) of about chunk_chars."""
        out, start, size = [], 0, 0
        for i, line in enumerate(self.lines):
            size += len(line) + 8
            if size >= self.chunk_chars:
                out.append((start, i + 1))
                start, size = i + 1, 0
        if start < len(self.lines):
            out.append((start, len(self.lines)))
        return out

    def _extract_range(self, s: int, e: int, depth: int = 0) -> list[RawTransaction]:
        """Each chunk OWNS lines [s, e) and also sees a little context around
        it, so chunks never overlap and nothing needs de-duplication. If the
        LLM fails on a chunk it is split in half and retried."""
        lo, hi = max(0, s - 3), min(len(self.lines), e + 8)
        text = "\n".join(f"L{i}: {self.lines[i]}" for i in range(lo, hi))
        try:
            rows = self._ask(
                TransactionExtraction, EXTRACT_PROMPT.format(start=s, end=e - 1, text=text), retries=1
            ).transactions
            return [r for r in rows if r.source_line is None or s <= r.source_line < e]
        except Exception:  # noqa: BLE001
            if e - s <= 4 or depth >= 3:
                raise
            mid = (s + e) // 2
            return self._extract_range(s, mid, depth + 1) + self._extract_range(mid, e, depth + 1)

    def extract_transactions(self):
        raw: list[RawTransaction] = []
        for s, e in self._ranges():
            try:
                raw += self._extract_range(s, e)
            except Exception as ex:  # noqa: BLE001
                self.warnings.append(f"Lines {s}-{e} failed extraction: {ex}")
        if not raw:
            raise RuntimeError("No transactions could be extracted")
        for n, r in enumerate(raw, 1):
            d = r.direction.strip().upper()
            direction = "DEBIT" if d.startswith(("D", "W")) else "CREDIT" if d.startswith(("C", "R")) else "UNKNOWN"
            self.tx.append(
                Transaction(
                    **{
                        **r.model_dump(),
                        "transaction_id": f"TXN_{n:05d}",
                        "amount": abs(r.amount),
                        "direction": direction,
                        "date": _iso(r.date),
                    }
                )
            )

    # ========================================================================
    # 3. VALIDATION  - dates, balances, totals, missing rows (pure math)
    # ========================================================================

    def validate(self):
        v = ValidationResult(transaction_count=len(self.tx))
        prev, checked, ok = self.account.opening_balance, 0, 0
        for t in self.tx:
            if t.balance is not None and prev is not None:
                delta = round(t.balance - prev, 2)
                if t.direction == "UNKNOWN":  # infer direction from the balance change
                    if abs(delta - t.amount) <= BALANCE_TOLERANCE:
                        t.direction = "CREDIT"
                    elif abs(delta + t.amount) <= BALANCE_TOLERANCE:
                        t.direction = "DEBIT"
                    if t.direction != "UNKNOWN":
                        v.directions_corrected += 1
                signed = t.amount if t.direction == "CREDIT" else -t.amount
                checked += 1
                if abs(delta - signed) <= BALANCE_TOLERANCE:
                    ok += 1
                elif abs(delta + signed) <= BALANCE_TOLERANCE and t.direction != "UNKNOWN":
                    # The running balance is the strongest evidence: flip the direction.
                    t.direction = "DEBIT" if t.direction == "CREDIT" else "CREDIT"
                    v.directions_corrected += 1
                    ok += 1
                else:
                    v.balance_break_transaction_ids.append(t.transaction_id)
            if t.balance is not None:
                prev = t.balance

        v.balance_consistency_percentage = round(100 * ok / checked, 2) if checked else 100.0
        tot = _totals(self.tx)
        v.credit_total, v.debit_total = tot["credit"], tot["debit"]

        if v.balance_break_transaction_ids:
            v.issues.append(
                f"{len(v.balance_break_transaction_ids)} running-balance breaks (rows may be missing or misread)"
            )
        if any(t.direction == "UNKNOWN" or not t.date for t in self.tx):
            v.issues.append("Some rows have unknown direction or date")
        last = self.tx[-1].balance if self.tx else None
        if self.account.closing_balance is not None and last is not None:
            if abs(self.account.closing_balance - last) > 0.5:
                v.issues.append("Last balance differs from closing balance (rows may be missing)")
        v.validation_status = (
            "FAIL" if v.balance_consistency_percentage < 70 else "PASS" if not v.issues else "WARNING"
        )
        self.validation = v

    # ========================================================================
    # 4. NORMALIZATION + RELATIONSHIP + ENTITY RESOLUTION
    # ========================================================================

    def normalize_and_resolve(self):
        """Batches run one after another so each batch can see the counterparties
        found so far (historical records, LOGIC §9)."""
        for start in range(0, len(self.tx), self.batch_size):
            batch = self.tx[start : start + self.batch_size]
            readings = self._read_narrations(batch, start // self.batch_size + 1)
            for t in batch:
                n = self._normalize(t, readings.get(t.transaction_id))
                self.norm[t.transaction_id] = n
                self.res[t.transaction_id] = self._resolve(t, n)

    def _read_narrations(self, batch: list[Transaction], batch_no: int) -> dict[str, NarrationReading]:
        """LLM: read names, type and relationship from the narrations."""
        top = sorted(self.cps.items(), key=lambda kv: -kv[1]["count"])[:40]
        existing = "; ".join(f"{k}: {v['name']}" for k, v in top) or "none"
        items = "\n".join(f"{t.transaction_id} | {t.direction} {t.amount} | {t.description}" for t in batch)
        prompt = ENRICH_PROMPT.format(
            holder=self.account.account_holder_name or "unknown",
            account=self.account.account_number or "unknown",
            modes=", ".join(PAYMENT_MODES),
            types=", ".join(COUNTERPARTY_TYPES),
            relationships=", ".join(RELATIONSHIPS),
            existing=existing,
            items=items,
        )
        try:
            return {o.transaction_id: o for o in self._ask(NarrationBatch, prompt).items}
        except Exception as e:  # noqa: BLE001
            self.warnings.append(f"Narration batch {batch_no} failed: {e}")
            return {}

    def _normalize(self, t: Transaction, reading: Optional[NarrationReading]) -> Normalized:
        """Normalization (GUIDE §2): derive structured fields, keep the raw row untouched.

        Deterministic (regex): payment mode, IFSC/bank code, UPI id, reference/UTR.
        LLM: names, alias, account, merchant, type, relationship.
        """
        r = reading or NarrationReading(transaction_id=t.transaction_id)
        text = t.description
        upper = text.upper()

        mode = next((m for m, p in _MODE_PATTERNS if re.search(p, upper)), None) or _enum(
            r.payment_mode, PAYMENT_MODES
        )
        ifsc = _IFSC_RE.search(upper)
        upi = _UPI_RE.search(text)
        ref = _REF_RE.search(text)
        name = r.counterparty_name_raw if _grounded(r.counterparty_name_raw, text) else None
        alias = r.counterparty_alias if _grounded(r.counterparty_alias, text) else None
        key = _key(name)

        # Internal transfer (LOGIC §15): needs real evidence (own account number,
        # own name, or the LLM saw "self"). Similar amounts are never evidence.
        evidence = list(r.evidence[:4])
        own = r.own_account_transfer_suspected
        acct_digits = _digits(self.account.account_number)
        if acct_digits and _digits(r.counterparty_account) == acct_digits:
            own = True
            evidence.append("counterparty account equals the holder's own account")
        if key and key == _key(self.account.account_holder_name):
            own = True
            evidence.append("counterparty name equals the account holder's name")

        return Normalized(
            **{
                **r.model_dump(),
                "payment_mode": mode,
                "counterparty_name_raw": name,
                "counterparty_alias": alias,
                "counterparty_type": _enum(r.counterparty_type, COUNTERPARTY_TYPES),
                "relationship": _enum(r.relationship, RELATIONSHIPS),
                "own_account_transfer_suspected": own,
                "evidence": evidence,
                "reference_number": t.reference_number or (ref.group(0) if ref else None),
                "bank_ifsc": ifsc.group(0) if ifsc else None,
                "upi_id": upi.group(0).lower() if upi else None,
                "normalized_name": key,
            }
        )

    # -- entity resolution (LOGIC §3, §4, §5) ----------------------------------

    @staticmethod
    def _identifiers(n: Normalized) -> list[str]:
        """Identity signals of a transaction, prefixed by kind."""
        ids = []
        if n.counterparty_account:
            ids.append(f"acct:{n.counterparty_account.strip()}")
        if n.upi_id:
            ids.append(f"upi:{n.upi_id}")
        if n.merchant_id:
            ids.append(f"mid:{_key(n.merchant_id)}")
        if n.bank_ifsc:
            ids.append(f"ifsc:{n.bank_ifsc}")
        if n.reference_number:
            ids.append(f"ref:{n.reference_number}")
        return ids

    @staticmethod
    def _conflict(cp: dict, ids: list[str]) -> bool:
        """Two different account numbers (or UPI ids) mean two different parties."""
        for kind in ("acct:", "upi:"):
            mine = {i for i in ids if i.startswith(kind)}
            theirs = {i for i in cp["ids"] if i.startswith(kind)}
            if mine and theirs and not (mine & theirs):
                return True
        return False

    def _score(self, cp: dict, ids: list[str], key: str, t: Transaction, n: Normalized, strong: bool) -> float:
        """LOGIC §5 confidence model: sum of the evidence weights, capped at 1."""
        s = sum(
            WEIGHTS[w] for kind, w in EXACT_KINDS.items() if any(i.startswith(kind) and i in cp["ids"] for i in ids)
        )
        s += WEIGHTS["strong_id"] if strong else 0
        s += WEIGHTS["name"] if _names_similar(key, cp["names"]) else 0
        s += WEIGHTS["narration"] if _narration_pattern(t.description) in cp["patterns"] else 0
        s += WEIGHTS["history"] if t.direction in cp["directions"] and n.payment_mode in cp["modes"] else 0
        return round(min(s, 1.0), 2)

    def _new_cp(self, name: str, ctype: str) -> str:
        cp_id = f"CP_{len(self.cps) + 1:05d}"
        self.cps[cp_id] = {
            "name": name, "type": ctype, "ids": set(), "names": {_key(name)}, "aliases": set(),
            "modes": set(), "directions": set(), "patterns": set(), "count": 0,
        }  # fmt: skip
        return cp_id

    def _attach(self, cp_id: str, t: Transaction, n: Normalized, ids: list[str]):
        """Record this transaction's signals on the counterparty (its history)."""
        cp = self.cps[cp_id]
        cp["ids"].update(ids)
        for i in ids:
            if i.startswith(tuple(EXACT_KINDS)):
                self._id_index.setdefault(i, cp_id)
        cp["names"].update(x for x in (n.normalized_name, _key(n.counterparty_alias)) if x)
        if n.counterparty_alias:
            cp["aliases"].add(n.counterparty_alias)
        cp["modes"].add(n.payment_mode)
        cp["directions"].add(t.direction)
        cp["patterns"].add(_narration_pattern(t.description))
        cp["count"] += 1
        if cp["type"] == "UNKNOWN":
            cp["type"] = n.counterparty_type

    def _resolve(self, t: Transaction, n: Normalized) -> Resolution:
        """LOGIC §4 decision ladder. The LLM proposes facts; this function
        decides identity.

            1. exact account number                    -> EXACT
            2. exact UPI id                            -> EXACT
            3. exact merchant / platform id            -> EXACT
            4. strong identifier (IFSC / reference)
               + compatible name, no conflicting id    -> STRONG
            5. similar name + same payment pattern     -> POSSIBLE (kept separate, review)
            6. otherwise                               -> NEW (or UNKNOWN if no identity at all)

        Strictness note: a name that repeats with no identifier at all (for
        example cash or cheque rows) lands on rule 5, so it is kept separate and
        flagged for the CA, as LOGIC §3 requires ("name similarity alone must
        not merge two entities").
        """
        ids = self._identifiers(n)
        key = n.normalized_name
        display = n.counterparty_name_raw or (n.merchant_id or "").upper() or None
        r = Resolution(
            transaction_id=t.transaction_id,
            counterparty_name_raw=n.counterparty_name_raw,
            counterparty_name_normalized=key or None,
            counterparty_type=n.counterparty_type,
            evidence=list(n.evidence[:4]),
        )
        cp_id, match, conf = None, "UNKNOWN", 0.0

        # Rules 1-3: exact identifiers, in evidence-priority order.
        for kind in EXACT_KINDS:
            hit = next((i for i in ids if i.startswith(kind) and i in self._id_index), None)
            if hit:
                cp_id, match = self._id_index[hit], "EXACT"
                r.evidence.append(f"exact match on {hit}")
                conf = self._score(self.cps[cp_id], ids, key, t, n, strong=False)
                break

        # Rule 4: strong identifier + compatible name.
        if cp_id is None:
            for cid, cp in self.cps.items():
                shared = [i for i in ids if i.startswith(STRONG_KINDS) and i in cp["ids"]]
                if shared and _names_similar(key, cp["names"]) and not self._conflict(cp, ids):
                    cp_id, match = cid, "STRONG"
                    r.evidence.append(f"strong identifier {shared[0]} with compatible name")
                    conf = self._score(cp, ids, key, t, n, strong=True)
                    break

        # Rules 5-6: possible match, or a new counterparty.
        if cp_id is None:
            possible = next(
                (
                    cid
                    for cid, cp in self.cps.items()
                    if _names_similar(key, cp["names"])
                    and t.direction in cp["directions"]
                    and n.payment_mode in cp["modes"]
                ),
                None,
            )
            # LOGIC §3 item 8: LLM judgment for unresolved cases. Still only POSSIBLE.
            possible = possible or (n.possible_existing_id if n.possible_existing_id in self.cps else None)
            identity = display or next((i.split(":", 1)[1] for i in ids if i.startswith(tuple(EXACT_KINDS))), None)
            if identity:
                cp_id, match, conf = self._new_cp(identity, n.counterparty_type), "NEW", n.confidence
                if possible:
                    match, r.possible_match_of, r.requires_review = "POSSIBLE", possible, True
                    conf = self._score(self.cps[possible], ids, key, t, n, strong=False)
                    r.evidence.append(f"possibly the same party as {possible}; kept separate")

        if cp_id:
            self._attach(cp_id, t, n, ids)
            cp = self.cps[cp_id]
            r.counterparty_id, r.counterparty_name, r.counterparty_type = cp_id, cp["name"], cp["type"]
        else:
            r.requires_review = True  # insufficient evidence -> UNKNOWN (LOGIC §9)
        r.match_type, r.confidence = match, round(conf, 2)
        return r

    # ========================================================================
    # 5. DUPLICATES  (LOGIC §14) - flag only, never delete
    # ========================================================================

    def detect_duplicates(self):
        """
            same reference / UTR (same amount + direction) -> HIGH
            same date + amount + direction + counterparty  -> MEDIUM
            same date + amount + direction + very similar narration -> POSSIBLE

        Each level skips transactions already flagged at a stronger level.
        Every original transaction is kept; only a duplicate_group_id links them.
        """
        count = 0

        def flag(ids: list[str], level: str):
            nonlocal count
            ids = [i for i in ids if i not in self.dup]
            if len(ids) > 1:
                count += 1
                for i in ids:
                    self.dup[i] = {"duplicate_group_id": f"DUP_{count:04d}", "confidence": level}

        by_ref, by_cp, by_day = defaultdict(list), defaultdict(list), defaultdict(list)
        for t in self.tx:
            tid, n, r = t.transaction_id, self.norm[t.transaction_id], self.res[t.transaction_id]
            if n.reference_number:
                by_ref[(n.reference_number, t.amount, t.direction)].append(tid)
            if t.date and r.counterparty_id:
                by_cp[(t.date, t.amount, t.direction, r.counterparty_id)].append(tid)
            if t.date:
                by_day[(t.date, t.amount, t.direction)].append(t)
        for ids in by_ref.values():
            flag(ids, "HIGH")
        for ids in by_cp.values():
            flag(ids, "MEDIUM")
        for ts in by_day.values():
            for a in ts:
                if a.transaction_id in self.dup:
                    continue
                similar = [
                    b.transaction_id
                    for b in ts
                    if b is not a
                    and b.transaction_id not in self.dup
                    and SequenceMatcher(None, a.description.upper(), b.description.upper()).ratio()
                    >= DUPLICATE_NARRATION_SIMILARITY
                ]
                flag([a.transaction_id] + similar, "POSSIBLE")

    # ========================================================================
    # 6. CLUSTERS  (LOGIC §6, GUIDE §4-5) - links only
    # ========================================================================

    @staticmethod
    def _recurring(ts: list[Transaction]) -> bool:
        """Recurring = at least 3 distinct months with a roughly monthly gap."""
        ds = sorted(datetime.fromisoformat(t.date) for t in ts if t.date)
        gaps = [(b - a).days for a, b in zip(ds, ds[1:]) if (b - a).days > 0]
        return len({d.strftime("%Y-%m") for d in ds}) >= 3 and bool(gaps) and 24 <= statistics.median(gaps) <= 36

    @staticmethod
    def _cluster_type(ctype: str, top_relationship: str, recurring: bool) -> str:
        if top_relationship == "WALLET_TOP_UP":
            return "WALLET"  # e.g. PAYTM ADD MONEY (GUIDE §5)
        if top_relationship == "TAX_PAYMENT":
            return "TAX"
        if ctype == "PLATFORM":
            return "MERCHANT"
        return "RECURRING_PAYMENT" if recurring else "COUNTERPARTY"

    def build_clusters(self):
        """One cluster per counterparty. The counterparty was already resolved
        with strict evidence rules, so a cluster never comes from weak name
        similarity. POSSIBLE matches are only listed as related clusters."""
        self.by_cp = defaultdict(list)
        for t in self.tx:
            cp_id = self.res[t.transaction_id].counterparty_id
            if cp_id:
                self.by_cp[cp_id].append(t)

        for n, (cp_id, ts) in enumerate(sorted(self.by_cp.items()), 1):
            cluster_id = f"CL_{n:04d}"
            cp = self.cps[cp_id]
            rs = [self.res[t.transaction_id] for t in ts]
            tot, (first, last) = _totals(ts), _date_range(ts)
            recurring = self._recurring(ts)
            top_rel = Counter(self.norm[t.transaction_id].relationship for t in ts).most_common(1)[0][0]
            evidence = [f"{c} transaction(s) matched as {m}" for m, c in Counter(r.match_type for r in rs).items()]
            evidence += sorted(i for i in cp["ids"] if not i.startswith("ref:"))
            self.clusters.append(
                EntityCluster(
                    cluster_id=cluster_id,
                    cluster_type=self._cluster_type(cp["type"], top_rel, recurring),
                    cluster_name=cp["name"],
                    counterparty_id=cp_id,
                    transaction_ids=[t.transaction_id for t in ts],
                    transaction_count=tot["count"],
                    total_credit=tot["credit"],
                    total_debit=tot["debit"],
                    net_amount=tot["net"],
                    first_transaction_date=first,
                    last_transaction_date=last,
                    recurring=recurring,
                    cluster_confidence=round(sum(r.confidence for r in rs) / len(rs), 2),
                    evidence=evidence,
                    possible_related_cluster_ids=sorted({r.possible_match_of for r in rs if r.possible_match_of}),
                    requires_review=any(r.requires_review for r in rs),
                )
            )
            for t in ts:
                self.cluster_of[t.transaction_id] = cluster_id
        by_cp_cluster = {c.counterparty_id: c.cluster_id for c in self.clusters}
        for c in self.clusters:  # counterparty ids -> cluster ids
            c.possible_related_cluster_ids = [
                by_cp_cluster[x] for x in c.possible_related_cluster_ids if x in by_cp_cluster
            ]

    # ========================================================================
    # 7. CLASSIFICATION  (LOGIC §11)  - after counterparty AND clusters
    # ========================================================================

    def classify(self):
        clusters = {c.cluster_id: c for c in self.clusters}
        for start in range(0, len(self.tx), self.batch_size):
            batch = self.tx[start : start + self.batch_size]
            rows = []
            for t in batch:
                r, n = self.res[t.transaction_id], self.norm[t.transaction_id]
                c = clusters.get(self.cluster_of.get(t.transaction_id, ""))
                ctx = (
                    f"cluster {c.cluster_type}: {c.transaction_count} txns, paid {c.total_debit}, "
                    f"received {c.total_credit}, recurring={c.recurring}"
                    if c
                    else "no cluster"
                )
                rows.append(
                    f"{t.transaction_id} | {t.date} | {t.direction} {t.amount} | mode={n.payment_mode} | "
                    f"rel={n.relationship} | party={r.counterparty_name} ({r.counterparty_type}) | {ctx} | {t.description}"
                )
            try:
                out = self._ask(
                    ClassificationBatch,
                    CLASSIFY_PROMPT.format(categories=", ".join(CATEGORIES), items="\n".join(rows)),
                ).items
            except Exception as e:  # noqa: BLE001
                self.warnings.append(f"Classification batch {start // self.batch_size + 1} failed: {e}")
                out = []
            got = {o.transaction_id: o for o in out}
            for t in batch:
                c = got.get(t.transaction_id) or Classification(transaction_id=t.transaction_id)
                c.category = _enum(c.category, CATEGORIES)
                # Guards: never guess the purpose (GUIDE §3, §8).
                weak = c.confidence < CLASSIFY_MIN_CONFIDENCE or not c.evidence
                wallet = self.norm[t.transaction_id].relationship == "WALLET_TOP_UP"  # not an expense (GUIDE §5)
                if weak or wallet:
                    c.category = "UNKNOWN"
                if c.category == "UNKNOWN" or c.confidence < REVIEW_LOW_CONFIDENCE:
                    c.requires_review = True
                self.cls[t.transaction_id] = c

    # ========================================================================
    # 8. CA REVIEW QUEUE  (LOGIC §17)
    # ========================================================================

    def build_review_queue(self):
        """Flag a transaction when: counterparty unknown, classification
        uncertain, entity match only possible, duplicate suspected, internal
        transfer suspected, personal/business unclear, balance does not
        reconcile, or extraction quality is poor."""

        def add(tid, kind, why, severity="MEDIUM"):
            self.review.append(ReviewItem(transaction_id=tid, issue_type=kind, reason=why, severity=severity))

        for tid in self.validation.balance_break_transaction_ids:
            add(tid, "BALANCE_MISMATCH", "Running balance does not reconcile with this row", "HIGH")
        for t in self.tx:
            tid = t.transaction_id
            r, n, c = self.res[tid], self.norm[tid], self.cls.get(tid)
            if t.extraction_confidence is not None and t.extraction_confidence < REVIEW_LOW_CONFIDENCE:
                add(tid, "POOR_EXTRACTION", "Low extraction confidence")
            if not r.counterparty_id:
                add(tid, "UNKNOWN_COUNTERPARTY", "Counterparty could not be established from the narration")
            elif r.match_type == "POSSIBLE":
                add(tid, "POSSIBLE_ENTITY_MATCH", f"Possibly the same party as {r.possible_match_of}; not merged")
            if c is None or c.requires_review or c.category == "UNKNOWN":
                add(
                    tid, "UNCERTAIN_CLASSIFICATION", f"Purpose not established ({c.category if c else 'unclassified'})"
                )
            elif c.category == "PERSONAL_EXPENSE" and c.confidence < PERSONAL_MIN_CONFIDENCE:
                add(tid, "PERSONAL_OR_BUSINESS_UNCLEAR", "Looks personal but the evidence is weak")
            if n.own_account_transfer_suspected:
                add(
                    tid,
                    "INTERNAL_TRANSFER_SUSPECTED",
                    "; ".join(n.evidence[:2]) or "Own-account evidence in narration",
                    "HIGH",
                )
            if tid in self.dup:
                d = self.dup[tid]
                add(
                    tid,
                    "DUPLICATE_SUSPECTED",
                    f"{d['confidence']} confidence ({d['duplicate_group_id']}); kept, not deleted",
                    "HIGH" if d["confidence"] == "HIGH" else "MEDIUM",
                )
        self.review.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x.severity])
        for item in self.review:
            self.flags[item.transaction_id].add(item.issue_type)

    # ========================================================================
    # 9. AGGREGATION  (LOGIC §12, §13, §16)
    # ========================================================================

    def _group_totals(self, keyfn) -> dict:
        buckets = defaultdict(list)
        for t in self.tx:
            buckets[keyfn(t)].append(t)
        return {k: _totals(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}

    def _side(self, direction: str) -> list[CounterpartySummary]:
        """'Who paid us?' (CREDIT) / 'Whom did we pay?' (DEBIT): group by resolved counterparty."""
        by = defaultdict(list)
        for t in self.tx:
            if t.direction == direction:
                by[self.res[t.transaction_id].counterparty_id].append(t)
        out = []
        for cp_id, ts in by.items():
            first, last = _date_range(ts)
            rs = [self.res[t.transaction_id] for t in ts]
            out.append(
                CounterpartySummary(
                    counterparty_id=cp_id,
                    counterparty=self.cps[cp_id]["name"] if cp_id else "UNRESOLVED",
                    cluster_id=self.cluster_of.get(ts[0].transaction_id),
                    transaction_count=len(ts),
                    total=round(sum(t.amount for t in ts), 2),
                    first_date=first,
                    last_date=last,
                    payment_modes=sorted({self.norm[t.transaction_id].payment_mode for t in ts}),
                    transaction_ids=[t.transaction_id for t in ts],
                    confidence=round(sum(r.confidence for r in rs) / len(rs), 2),
                    review_flags=sorted({f for t in ts for f in self.flags[t.transaction_id]}),
                )
            )
        return sorted(out, key=lambda s: -s.total)

    def _category_of(self, t: Transaction) -> str:
        c = self.cls.get(t.transaction_id)
        return c.category if c else "UNKNOWN"

    def aggregate(self):
        # Per counterparty (LOGIC §16). Per cluster totals live on EntityCluster.
        cluster_of_cp = {c.counterparty_id: c.cluster_id for c in self.clusters}
        for cp_id, cp in self.cps.items():
            ts = self.by_cp.get(cp_id, [])
            tot, (first, last) = _totals(ts), _date_range(ts)
            self.counterparties.append(
                Counterparty(
                    counterparty_id=cp_id,
                    name=cp["name"],
                    counterparty_type=cp["type"],
                    aliases=sorted(cp["aliases"]),
                    identifiers=sorted(i for i in cp["ids"] if not i.startswith("ref:")),
                    cluster_id=cluster_of_cp.get(cp_id),
                    transaction_count=tot["count"],
                    total_credit=tot["credit"],
                    total_debit=tot["debit"],
                    net_amount=tot["net"],
                    first_transaction_date=first,
                    last_transaction_date=last,
                )
            )
        self.received, self.paid = self._side("CREDIT"), self._side("DEBIT")

        # Separate receipts / payments by purpose "when evidence supports it" (LOGIC §12, §13).
        for side, direction in (("received", "CREDIT"), ("paid", "DEBIT")):
            by = defaultdict(float)
            for t in self.tx:
                if t.direction == direction:
                    by[self._category_of(t)] += t.amount
            self.category_breakdown[side] = {k: round(v, 2) for k, v in sorted(by.items())}

        self.categories = self._group_totals(self._category_of)
        self.modes = self._group_totals(lambda t: self.norm[t.transaction_id].payment_mode)
        self.months = self._group_totals(lambda t: _month(t.date) or "UNKNOWN")

    # ========================================================================
    # RESULT
    # ========================================================================

    def _flatten(self, t: Transaction) -> dict:
        """One transaction tree (LOGIC §18): raw, normalized, counterparty,
        cluster, classification, review flags. Gives the CA an audit trail
        from the dashboard back to the statement (LOGIC §19)."""
        tid = t.transaction_id
        return {
            "raw": t.model_dump(),
            "normalized": self.norm[tid].model_dump(exclude={"transaction_id"}),
            "counterparty": self.res[tid].model_dump(),
            "cluster_id": self.cluster_of.get(tid),
            "classification": self.cls[tid].model_dump() if tid in self.cls else None,
            "duplicate": self.dup.get(tid),
            "review_flags": sorted(self.flags[tid]),
        }

    def build_result(self) -> StatementResult:
        v = self.validation
        return StatementResult(
            account=self.account,
            transaction_count=len(self.tx),
            total_credit=v.credit_total,
            total_debit=v.debit_total,
            net_movement=round(v.credit_total - v.debit_total, 2),
            validation=v,
            money_received=self.received,
            money_paid=self.paid,
            category_breakdown=self.category_breakdown,
            counterparties=self.counterparties,
            clusters=self.clusters,
            categories=self.categories,
            payment_modes=self.modes,
            monthly_trends=self.months,
            unknown_transaction_ids=[tid for tid, c in self.cls.items() if c.category == "UNKNOWN"],
            review_queue=self.review,
            warnings=self.warnings,
            model=self.model_name,
            prompt_version=PROMPT_VERSION,
            transactions=[self._flatten(t) for t in self.tx] if self.include_transactions else None,
        )

    def run_all(self) -> StatementResult:
        if not self.lines:
            raise RuntimeError("Empty statement text")
        self._run("account", self.extract_account)
        self._run("extract", self.extract_transactions, fatal=True)
        self._run("validate", self.validate)
        self._run("counterparties", self.normalize_and_resolve)
        for t in self.tx:  # keep later stages safe if a stage above failed
            self.norm.setdefault(t.transaction_id, Normalized(transaction_id=t.transaction_id))
            self.res.setdefault(t.transaction_id, Resolution(transaction_id=t.transaction_id, requires_review=True))
        self._run("duplicates", self.detect_duplicates)
        self._run("clusters", self.build_clusters)
        self._run("classify", self.classify)
        for t in self.tx:
            self.cls.setdefault(
                t.transaction_id, Classification(transaction_id=t.transaction_id, requires_review=True)
            )
        self._run("review", self.build_review_queue)
        self._run("aggregate", self.aggregate)
        return self.build_result()


# ============================================================
# PUBLIC API
# ============================================================


def process_statement_v2(data: str, group_by=None, include_transactions=False, use_agent=False) -> StatementResult:
    """Process the text of one bank statement.

    `group_by` and `use_agent` are accepted only so existing callers keep
    working; they are ignored. The result always contains every grouping the
    design documents ask for (counterparty, cluster, category, month, payment
    mode), so no group selection is needed.
    """
    llm = get_ollama_instance()
    return StatementPipeline(data, llm, include_transactions=include_transactions).run_all()
