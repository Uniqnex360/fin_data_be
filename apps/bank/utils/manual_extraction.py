import re
import pymupdf


def extract_pdf_pages(pdf_file):
    """
    Extract raw text from every PDF page.

    Returns:
        {
            "page_1": "...",
            "page_2": "...",
            ...
        }
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


import re
from decimal import Decimal

DATE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def clean_amount(value):
    if not value or value.strip() == "—":
        return Decimal("0")

    value = value.strip()

    # Your sample uses I instead of ₹
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

            # Transaction always starts with a date
            if not DATE_PATTERN.match(lines[i]):
                i += 1
                continue

            # Need at least the 7 expected fields
            if i + 6 >= len(lines):
                break

            date = lines[i]
            narration = lines[i + 1]
            reference = lines[i + 2]
            value_date = lines[i + 3]
            withdrawal = lines[i + 4]
            deposit = lines[i + 5]
            closing_balance = lines[i + 6]

            # Make sure this really looks like a transaction
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


import re


def normalize_narration(narration):
    if not narration:
        return ""

    value = narration.upper()

    # Normalize separators
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    # Remove unnecessary whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_transactions(transactions):

    for transaction in transactions:

        transaction["normalized_narration"] = normalize_narration(transaction["narration"])

        transaction["transaction_type"] = "CREDIT" if transaction["deposit"] > 0 else "DEBIT"

        transaction["amount"] = transaction["deposit"] if transaction["deposit"] > 0 else transaction["withdrawal"]

    return transactions


from collections import defaultdict


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

    # 1. PDF → raw text
    pages = extract_pdf_pages(pdf_file)

    # 2. Account-level information
    account = extract_account_details(pages)

    # 3. Raw transactions
    transactions = parse_transactions(pages)

    # 4. Normalize
    transactions = normalize_transactions(transactions)

    # 5. Cluster
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


from apps.common.views import AppAPIView



class ManualDataExtractionView(AppAPIView):

    permission_classes = []

    def post(self, request, *args, **kwargs):

        pdf_file = request.FILES.get("file")

        if not pdf_file:
            return self.send_response(message="PDF file is required.")

        try:

            result = process_bank_statement(pdf_file)

            return self.send_response(data=result)

        except Exception as exc:

            return self.send_response(message=f"Failed to process PDF: {str(exc)}")
