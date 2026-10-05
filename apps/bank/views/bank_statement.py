import mimetypes
import pymupdf

from langchain_ollama import ChatOllama

from apps.common.views import get_upload_api_view, AppAPIView
from apps.common.llms import get_ollama_instance
from apps.bank.models import BankDocumentFile
from apps.bank.utils import extract_bank_statement_data

BankFilesUploadAPIView = get_upload_api_view(meta_model=BankDocumentFile)


def get_document_obj(id: int) -> BankDocumentFile | None:
    return BankDocumentFile.objects.filter(id=id).first()


def get_file_meta_fields(file) -> dict:
    mime_type, encoding = mimetypes.guess_type(file.name)

    return {
        "name": file.name,
        "size": file.size,
        "mime_type": mime_type,
        "encoding": encoding,
    }


def handle_pdf(file, llm) -> dict:
    """
    Extract text from every PDF page and use the LLM
    to convert the raw bank statement text into structured data.
    """

    all_transactions = []
    account_details = {}
    summary = {}
    reward_points = []
    errors = []

    with file.open("rb") as f:
        pdf_bytes = f.read()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    try:
        print("Number of pages:", len(doc))

        # for page_index in range(len(doc)):
        for page_index in range(1):
            page = doc[page_index]
            page_number = page_index + 1

            text = page.get_text("text", sort=True).strip()

            if not text:
                print(f"Skipping empty page: {page_number}")
                continue

            print(f"\n========== PAGE {page_number} ==========")

            prompt = f"""
You are extracting data from page {page_number} of an Indian bank statement.
Use ONLY the statement text below and copy values exactly from it.
Never use placeholder or example values. If a value is not in the text, leave it null.

Rules:
- Extract the account details from the header.
- Return EVERY transaction row on the page, in order, including the opening BALANCE FORWARD row.
- Each row has: date, value date, description (may span several lines), an optional deposit or withdrawal amount, and a balance.
- The balance is the last amount of a row. Each row has at most one amount besides the balance.
- Compare the row's balance with the previous row's balance:
  balance went down -> DEBIT, amount goes in debit_amount;
  balance went up -> CREDIT, amount goes in credit_amount.
- Dates as YYYY-MM-DD. Numbers without commas.
- A row with no date shown uses the date of the row above it.
- Preserve raw_narration exactly, only joining multiline text with spaces.
- Extract UTR/reference numbers when present.
- transaction_category must be UNKNOWN when it cannot be determined confidently.
- Do not classify normal merchant payments as UTILITY unless the narration clearly indicates a utility payment.

Statement text:

{text}
"""

            try:
                response = llm.invoke(prompt)
            except Exception as e:
                print(f"Page {page_number} failed:", str(e))
                errors.append({"page": page_number, "error": str(e)})
                continue

            print("LLM response:", response)

            # account: later pages only fill values that are still empty
            for key, value in response.account.model_dump().items():
                if value is not None and account_details.get(key) is None:
                    account_details[key] = value

            # summary: same rule
            if response.summary:
                for key, value in response.summary.model_dump().items():
                    if value is not None:
                        summary[key] = value

            for transaction in response.transactions:
                transaction.source_page = page_number
                all_transactions.append(transaction.model_dump())  # all keys, nulls included

            reward_points.extend(rp.model_dump() for rp in response.reward_points)

    finally:
        doc.close()

    return {
        "account": account_details,
        "transactions": all_transactions,
        "summary": summary,
        "reward_points": reward_points,
        "errors": errors,
    }


class BankDocsAPIView(AppAPIView):

    def post(self, request):

        document = request.data.get("document")

        file_obj: BankDocumentFile | None = get_document_obj(document)

        if file_obj is None:
            return self.send_error_response({"detail": "File is Not found"})

        file_meta = get_file_meta_fields(file_obj.file)

        mime_type = file_meta.get("mime_type")

        from apps.bank.utils.statement import BankStatementExtraction

        llm: ChatOllama = get_ollama_instance()

        structured_llm = llm.with_structured_output(BankStatementExtraction, method="json_schema")

        data = None

        if mime_type == "application/pdf":

            data = handle_pdf(
                file_obj.file,
                structured_llm,
            )

        return self.send_response(
            {
                "document": document,
                "data": data,
            }
        )


    
