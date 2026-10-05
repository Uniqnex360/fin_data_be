from typing import Optional, Literal, List
from pydantic import BaseModel, Field

from langchain.tools import tool


class Transaction(BaseModel):
    transaction_date: Optional[str] = Field(None, description="The row's main date, YYYY-MM-DD. If the row shows no date, use the date of the row above.")
    value_date: Optional[str] = Field(None, description="The row's value date if the table has one, YYYY-MM-DD. Otherwise same as transaction_date.")
    transaction_id: Optional[str] = Field(None, description="Transaction/serial id if the row has one, else the main reference number in the narration.")
    utr_reference: Optional[str] = Field(None, description="UTR/RRN/UPI reference: the long numeric reference in the narration.")
    cheque_number: Optional[str] = Field(None, description="Cheque number if the row has a cheque/instrument column value.")
    raw_narration: str = Field(..., description="Full description of this one row, multiline joined with spaces, unchanged.")
    is_multiline: bool = Field(False, description="True if the description spanned more than one line.")
    debit_amount: Optional[float] = Field(None, description="Money going out. Set when the balance went down.")
    credit_amount: Optional[float] = Field(None, description="Money coming in. Set when the balance went up.")
    running_balance: Optional[float] = Field(None, description="Balance after this row, the row's last number. Always fill.")

    direction: Optional[Literal["DEBIT", "CREDIT"]] = Field(
        None, description="DEBIT if money went out, CREDIT if money came in. Decide from the amount column or balance change.")

    transaction_category: Literal[
        "SALARY", "EMI", "RENT", "TAX", "UTILITY", "VENDOR_PAYMENT",
        "CUSTOMER_RECEIPT", "LOAN", "INTEREST", "BANK_CHARGE", "INVESTMENT",
        "TRANSFER", "CASH", "REFUND", "DIRECTOR", "SHAREHOLDER", "OTHER", "UNKNOWN",
    ] = Field("UNKNOWN", description="Infer the purpose from the narration. Cash withdrawals/deposits = CASH, person-to-person or own-account moves = TRANSFER, payments to shops/companies = VENDOR_PAYMENT, credits from customers = CUSTOMER_RECEIPT.")

    counterparty_name: Optional[str] = Field(None, description="The merchant, person or company named in the narration. For own ATM withdrawals use 'SELF'.")
    counterparty_account_reference: Optional[str] = Field(None, description="UPI id, VPA or account number of the other party found in the narration.")
    counterparty_bank_ifsc_or_code: Optional[str] = Field(None, description="IFSC or bank code of the other party found in the narration.")

    payment_mode: Literal[
        "NEFT", "RTGS", "IMPS", "UPI", "CHEQUE", "CASH", "CARD", "ATM_WITHDRAWAL",
        "ECS", "NACH", "TRANSFER", "BANK_CHARGE", "TAX_ADJUSTMENT", "INTEREST",
        "OTHER", "UNKNOWN",
    ] = Field("UNKNOWN", description="Infer from narration keywords: UPI, NEFT, RTGS, IMPS, ATM/ATM WITHDRAWAL, POS/PURCHASE = CARD, CHQ/CHEQUE, etc.")

    normalized_narration: Optional[str] = Field(None, description="Cleaned short readable version of the narration: no ids or long numbers, e.g. 'UPI payment to <name>'.")
    references: List[str] = Field(default_factory=list, description="Every reference number or id found in the narration.")

    entity_type: Literal[
        "VENDOR", "CUSTOMER", "EMPLOYEE", "DIRECTOR", "SHAREHOLDER", "BANK",
        "LENDER", "GOVERNMENT", "SELF", "MERCHANT", "UNKNOWN",
    ] = Field("UNKNOWN", description="Who the counterparty is. Own ATM withdrawal = SELF, shops/apps/online services = MERCHANT, bank fees = BANK.")

    is_charge_or_tax_line: bool = Field(False, description="True if this row is a bank charge, fee, GST or tax.")
    charge_type: Optional[Literal["CGST", "SGST", "IGST", "BANK_FEE", "OTHER"]] = Field(None, description="Only when is_charge_or_tax_line is true.")

    linked_transaction_id: Optional[str] = Field(None, description="Reference of the original transaction if this row is a reversal, refund or adjustment of it.")
    source_file: Optional[str] = None
    source_page: Optional[int] = None
    source_row: Optional[int] = None
    ocr_text: Optional[str] = None


class AccountDetails(BaseModel):
    bank_name: Optional[str] = Field(None, description="Bank name. If not printed, infer from the IFSC prefix or logo text.")
    account_holder_name: Optional[str] = Field(None, description="Name of the customer at the top of the statement.")
    account_holder_address: Optional[str] = Field(None, description="Customer address lines joined with commas.")
    account_number: Optional[str] = Field(None, description="Account number as printed.")
    masked_account: Optional[str] = Field(None, description="Account number with all but the last 4 digits as X.")
    branch: Optional[str] = Field(None, description="Branch name.")
    branch_address: Optional[str] = Field(None, description="Branch address.")
    ifsc: Optional[str] = Field(None, description="IFSC code.")
    micr: Optional[str] = Field(None, description="MICR code.")
    branch_phone: Optional[str] = Field(None, description="Branch phone number.")
    account_type: Optional[str] = Field(None, description="Account type, e.g. savings or current.")
    currency: Optional[str] = Field(None, description="Currency code. INR if amounts are in rupees.")
    statement_date: Optional[str] = Field(None, description="Statement date, YYYY-MM-DD.")
    statement_period: Optional[str] = Field(None, description="'from to to' dates. If not printed, use the first and last transaction dates.")
    nominee_registered: Optional[bool] = Field(None, description="True/false if the statement says whether a nominee is registered.")


class StatementSummary(BaseModel):
    total_deposits: Optional[float] = Field(None, description="Sum of all credits. Use the printed total if present, else add up credit_amount of the transactions on this page.")
    total_withdrawals: Optional[float] = Field(None, description="Sum of all debits. Use the printed total if present, else add up debit_amount.")
    closing_balance: Optional[float] = Field(None, description="Balance of the last row on the page.")
    opening_balance: Optional[float] = Field(None, description="Opening/brought-forward balance, or balance of the first row.")


class RewardPointsEntry(BaseModel):
    scheme_name: Optional[str] = None
    opening_balance: Optional[float] = None
    points_accrued: Optional[float] = None
    points_redeemed: Optional[float] = None
    adjustment_bonus: Optional[float] = None
    closing_balance: Optional[float] = None


class BankStatementExtraction(BaseModel):
    account: AccountDetails
    transactions: List[Transaction] = Field(default_factory=list)
    summary: Optional[StatementSummary] = None
    reward_points: List[RewardPointsEntry] = Field(default_factory=list)

@tool
def extract_bank_statement_data(statement_text: str) -> dict:
    """
    Extract structured bank account and transaction data from raw
    bank statement text.
    """

    return {
        "account": {},
        "transactions": [],
        "summary": {},
        "reward_points": [],
    }
