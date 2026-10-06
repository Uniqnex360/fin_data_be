import pymupdf

from apps.common.views import AppAPIView, AppModelListAPIViewSet
from apps.bank.models import BankDocumentFile, BankAccountBase
from apps.bank.serializers import BankDocumentFileSerializer

from .bank_statement import get_document_obj, get_file_meta_fields
from apps.common.llms import get_ollama_instance
from apps.bank.utils.cluster import process_statement
from apps.bank.utils import process_statement_v2, save_cluster_v2_result

# ----------------------------------------------------------------------------------------
# View
# ----------------------------------------------------------------------------------------


class ClusterBankStatementAPIView(AppAPIView):

    def post(self, request):

        document = request.data.get("document")

        # optional, e.g. "category,counterparty,month" or ["category", "month"]; credit/debit is always a group key
        group_by = request.data.get("group_by")
        if isinstance(group_by, str):
            group_by = [g.strip() for g in group_by.split(",") if g.strip()]
        include_transactions = str(request.data.get("include_transactions", "")).lower() in ("1", "true", "yes")

        file_obj: BankDocumentFile = get_document_obj(document)

        if file_obj is None:
            return self.send_error_response({"detail": "File is Not found"})

        file_meta = get_file_meta_fields(file_obj.file)

        mime_type = file_meta.get("mime_type")

        data = ""

        if mime_type == "application/pdf":

            with file_obj.file.open("rb") as f:

                pdf_bytes = f.read()

            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            pages = []
            try:
                for page_index in range(len(doc)):

                    page = doc[page_index]
                    pages.append(page.get_text("text", sort=True).strip())
            finally:
                doc.close()

            # join pages with a newline, otherwise the last line of a page glues onto the first line of the next
            data = "\n".join(pages)
        else:
            return self.send_error_response({"detail": f"Unsupported file type: {mime_type}"})

        if not data.strip():
            return self.send_error_response({"detail": "No text found in the PDF (scanned file? run OCR first)"})

        try:
            result = process_statement(data, group_by=group_by, include_transactions=include_transactions)
        except Exception as e:
            print("Bank statement clustering failed")
            return self.send_error_response({"detail": f"Could not process the statement: {e}"})

        return self.send_response(result.model_dump())


class ClusterBankStatementAPIViewV2(AppAPIView):

    permission_classes = []

    def post(self, request):

        document = request.data.get("document")

        # optional, e.g. "category,counterparty,month" or ["category", "month"]; credit/debit is always a group key
        group_by = request.data.get("group_by")
        if isinstance(group_by, str):
            group_by = [g.strip() for g in group_by.split(",") if g.strip()]
        include_transactions = str(request.data.get("include_transactions", "")).lower() in ("1", "true", "yes")

        file_obj: BankDocumentFile = get_document_obj(document)

        if file_obj is None:
            return self.send_error_response({"detail": "File is Not found"})

        file_meta = get_file_meta_fields(file_obj.file)

        mime_type = file_meta.get("mime_type")

        data = ""

        if mime_type == "application/pdf":

            with file_obj.file.open("rb") as f:

                pdf_bytes = f.read()

            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            pages = []
            try:
                for page_index in range(len(doc)):

                    page = doc[page_index]
                    pages.append(page.get_text("text", sort=True).strip())
            finally:
                doc.close()

            # join pages with a newline, otherwise the last line of a page glues onto the first line of the next
            data = "\n".join(pages)
        else:
            return self.send_error_response({"detail": f"Unsupported file type: {mime_type}"})

        if not data.strip():
            return self.send_error_response({"detail": "No text found in the PDF (scanned file? run OCR first)"})

        try:
            result = process_statement_v2(data, group_by=group_by, include_transactions=include_transactions)
            # saving the extracted result
            save_cluster_v2_result(result, document_id=int(document))
        except Exception as e:
            print("Bank statement clustering failed")
            return self.send_error_response({"detail": f"Could not process the statement: {e}"})

        return self.send_response(result.model_dump())


class BankFileListAPIViewSet(AppModelListAPIViewSet):

    queryset = BankDocumentFile.objects.all().order_by("-created")
    serializer_class = BankDocumentFileSerializer
    permission_classes = []

from apps.bank.models import BankAccountBase


class BankStatementDetailAPIView(AppAPIView):

    permission_classes = []

    def get(self, request, *args, **kwargs):
        document_id = kwargs.get("id")

        obj = (
            BankAccountBase.objects
            .filter(file_id=document_id)
            .order_by("-created")
            .first()
        )

        if obj is None:
            return self.send_error_response(
                {"error": "extracted data is not available"}
            )

        data = {
            # Account
            "bank_name": obj.bank_name,
            "account_holder_name": obj.account_holder_name,
            "account_number": obj.account_number,
            "ifsc": obj.ifsc,
            "statement_period_start": obj.statement_period_start,
            "statement_period_end": obj.statement_period_end,

            # Summary
            "opening_balance": obj.opening_balance,
            "closing_balance": obj.closing_balance,
            "transaction_count": obj.transaction_count,
            "total_credit": obj.total_credit,
            "total_debit": obj.total_debit,
            "net_movement": obj.net_movement,

            # Validation
            "validations": obj.validations,

            # Analysis
            "money_received": obj.money_received,
            "money_paid": obj.money_paid,
            "category_breakdown": obj.category_breakdown,
            "counterparties": obj.counterparties,
            "clusters": obj.clusters,
            "categories": obj.categories,
            "payment_modes": obj.payment_modes,
            "monthly_trends": obj.monthly_trends,

            # Other
            "unknown_transaction_ids": obj.unknown_transaction_ids,
            "review_queue": obj.review_queue,
            "warnings": obj.warnings,
            "model": obj.model,
            "prompt_version": obj.prompt_version,
            "transactions": obj.transactions,
        }

        return self.send_response(data)