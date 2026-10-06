from decimal import Decimal
from datetime import date

from apps.bank.models import BankAccountBase, BankDocumentFile


def save_cluster_v2_result(data, document_id: int) -> None:
    account = data.account
    validation = data.validation

    file = BankDocumentFile.objects.get(id=document_id)

    BankAccountBase.objects.create(
        file=file,
        # Account
        bank_name=account.bank_name,
        account_holder_name=account.account_holder_name,
        account_number=account.account_number,
        ifsc=account.ifsc,
        statement_period_start=(
            date.fromisoformat(account.statement_period_start) if account.statement_period_start else None
        ),
        statement_period_end=(
            date.fromisoformat(account.statement_period_end) if account.statement_period_end else None
        ),
        opening_balance=(Decimal(str(account.opening_balance)) if account.opening_balance is not None else None),
        closing_balance=(Decimal(str(account.closing_balance)) if account.closing_balance is not None else None),
        # Summary
        transaction_count=data.transaction_count,
        total_credit=Decimal(str(data.total_credit)),
        total_debit=Decimal(str(data.total_debit)),
        net_movement=Decimal(str(data.net_movement)),
        # Validation
        validations=validation.model_dump(),
        # Analytical results
        money_received=[item.model_dump() for item in data.money_received],
        money_paid=[item.model_dump() for item in data.money_paid],
        category_breakdown=data.category_breakdown,
        counterparties=[item.model_dump() for item in data.counterparties],
        clusters=[item.model_dump() for item in data.clusters],
        categories=data.categories,
        payment_modes=data.payment_modes,
        monthly_trends=data.monthly_trends,
        unknown_transaction_ids=data.unknown_transaction_ids,
        review_queue=[item.model_dump() for item in data.review_queue],
        warnings=data.warnings,
        # Processing info
        model=data.model,
        prompt_version=data.prompt_version,
        # Large transaction tree
        transactions=data.transactions,
    )
