import re

import pymupdf

from decimal import Decimal
from collections import defaultdict

from django.db import transaction

from apps.bank.models import ManualBankAccountBase
from apps.common.views import AppAPIView, AppModelListAPIViewSet
from apps.bank.serializers import ManualBankAccountBaseSerializer


def extract_pdf_pages(pdf_file):
    """
    Extract raw text from every PDF page.
    """
    pdf_document = pymupdf.open(
        stream=pdf_file.read(),
        filetype="pdf",
    )

    try:
        return {f"page_{page_number}": page.get_text() for page_number, page in enumerate(pdf_document, start=1)}
    finally:
        pdf_document.close()


def extract_account_details(pages):
    """
    Extract account-level information from the statement.
    """
    full_text = "\n".join(pages.values())

    def extract(pattern):
        match = re.search(pattern, full_text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    return {
        "business_name": extract(r"Business Name\s*\n(.+)"),
        "account_reference": extract(r"Account Reference\s*\n(.+)"),
        "business_type": extract(r"Business Type\s*\n(.+)"),
        "statement_period": extract(r"Statement Period\s*\n(.+)"),
        "opening_balance": extract(r"Opening Balance\s*\n([^\n]+)"),
        "closing_balance": extract(r"Closing Balance\s*\n([^\n]+)"),
    }


DATE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def clean_amount(value):
    if not value or value.strip() == "—":
        return Decimal("0")

    value = value.strip()
    value = value.replace("I", "")
    value = value.replace(",", "")
    value = value.replace("₹", "")

    return Decimal(value)


def parse_transactions(pages):
    """
    Convert extracted PDF text into structured transactions.
    """
    transactions = []

    for page_number, text in pages.items():
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        i = 0

        while i < len(lines):

            if not DATE_PATTERN.match(lines[i]):
                i += 1
                continue

            if i + 6 >= len(lines):
                break

            date = lines[i]
            narration = lines[i + 1]
            reference = lines[i + 2]
            value_date = lines[i + 3]
            withdrawal = lines[i + 4]
            deposit = lines[i + 5]
            closing_balance = lines[i + 6]

            if not DATE_PATTERN.match(value_date):
                i += 1
                continue

            transactions.append(
                {
                    "date": date,
                    "narration": narration,
                    "reference": reference,
                    "value_date": value_date,
                    "withdrawal": clean_amount(withdrawal),
                    "deposit": clean_amount(deposit),
                    "closing_balance": clean_amount(closing_balance),
                    "page": page_number,
                }
            )

            i += 7

    return transactions


def normalize_narration(narration):
    if not narration:
        return ""

    value = narration.upper()

    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_transactions(transactions):
    for transaction in transactions:

        transaction["normalized_narration"] = normalize_narration(transaction["narration"])

        transaction["transaction_type"] = "CREDIT" if transaction["deposit"] > 0 else "DEBIT"

        transaction["amount"] = transaction["deposit"] if transaction["deposit"] > 0 else transaction["withdrawal"]

    return transactions


def cluster_transactions(transactions):
    clusters = defaultdict(list)

    for transaction in transactions:
        key = (
            transaction["normalized_narration"],
            transaction["transaction_type"],
        )

        clusters[key].append(transaction)

    result = []

    for cluster_id, ((name, transaction_type), items) in enumerate(
        clusters.items(),
        start=1,
    ):
        total = sum(item["amount"] for item in items)

        result.append(
            {
                "cluster_id": cluster_id,
                "cluster_key": name,
                "transaction_type": transaction_type,
                "transaction_count": len(items),
                "total_amount": total,
                "transactions": items,
            }
        )

    return result


def process_bank_statement(pdf_file):
    """
    PDF -> extracted and processed bank statement data.
    """

    pages = extract_pdf_pages(pdf_file)

    account = extract_account_details(pages)

    transactions = parse_transactions(pages)

    transactions = normalize_transactions(transactions)

    clusters = cluster_transactions(transactions)

    return {
        "account": account,
        "transactions": transactions,
        "clusters": clusters,
        "summary": {
            "page_count": len(pages),
            "transaction_count": len(transactions),
            "cluster_count": len(clusters),
        },
    }


def make_json_serializable(value):
    """
    Convert Decimal values recursively so the result can be
    stored safely in JSONField.
    """

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    return value


@transaction.atomic
def save_bank_statement_result(result):
    """
    Persist extracted bank statement data.

    This function is responsible only for database persistence.
    """

    account = result["account"]

    opening_balance = account.get("opening_balance")
    closing_balance = account.get("closing_balance")

    account_data = {
        "business_name": account.get("business_name"),
        "account_reference": account.get("account_reference"),
        "business_type": account.get("business_type"),
        "statement_period": account.get("statement_period"),
        "opening_balance": (clean_amount(opening_balance) if opening_balance else None),
        "closing_balance": (clean_amount(closing_balance) if closing_balance else None),
        "transactions": make_json_serializable(result.get("transactions", [])),
        "clusters": make_json_serializable(result.get("clusters", [])),
    }

    return ManualBankAccountBase.objects.create(
        **account_data,
    )


class ManualDataExtractionView(AppAPIView):

    permission_classes = []

    def post(self, request, *args, **kwargs):

        pdf_file = request.FILES.get("file")

        if not pdf_file:
            return self.send_response(message="PDF file is required.")

        try:
            result = process_bank_statement(pdf_file)

            bank_account = save_bank_statement_result(result)

            return self.send_response(
                data={
                    "id": bank_account.id,
                    **result,
                }
            )

        except Exception as exc:
            return self.send_response(message=f"Failed to process PDF: {str(exc)}")


class ManualExtractionListAPIViewSet(AppModelListAPIViewSet):

    queryset = ManualBankAccountBase.objects.all().order_by("-created")
    serializer_class = ManualBankAccountBaseSerializer
    permission_classes = []


class ManualBankStatementDetailAPIView(AppAPIView):

    permission_classes = []

    def get(self, request, *args, **kwargs):
        id = kwargs.get("id")

        obj = ManualBankAccountBase.objects.filter(id=id).order_by("-created").first()

        if obj is None:
            return self.send_error_response({"error": "extracted data is not available"})

        data = {
            # Account
            "business_name": obj.business_name,
            "account_reference": obj.account_reference,
            "business_type": obj.business_type,
            "statement_period": obj.statement_period,
            # Summary
            "opening_balance": obj.opening_balance,
            "closing_balance": obj.closing_balance,
            # Transactions
            "transactions": obj.transactions,
            # Clusters
            "clusters": obj.clusters,
            # Processing information
            "model": obj.model,
            "prompt_version": obj.prompt_version,
        }

        return self.send_response(data)
