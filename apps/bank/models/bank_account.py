from django.db import models

from apps.common.models import (
    BaseFileUploadModel,
    UploadToFolder,
    BaseModel,
    COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    COMMON_CHAR_FIELD_MAX_LENGTH,
)


class BankDocumentFile(BaseFileUploadModel):
    file = models.FileField(
        upload_to=UploadToFolder("bank_document_file"),
        max_length=500,
    )


class BankAccountBase(BaseModel):
    file = models.ForeignKey(
        BankDocumentFile,
        on_delete=models.CASCADE,
    )

    # Account summary
    bank_name = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    account_holder_name = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    account_number = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    ifsc = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )

    statement_period_start = models.DateField(
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    statement_period_end = models.DateField(
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )

    opening_balance = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    closing_balance = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )

    transaction_count = models.PositiveIntegerField(
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )

    total_credit = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    total_debit = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    net_movement = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )

    # Validation
    validations = models.JSONField(default=dict)

    # Analytical results
    money_received = models.JSONField(default=list)
    money_paid = models.JSONField(default=list)

    category_breakdown = models.JSONField(default=dict)
    counterparties = models.JSONField(default=list)
    clusters = models.JSONField(default=list)

    categories = models.JSONField(default=dict)
    payment_modes = models.JSONField(default=dict)
    monthly_trends = models.JSONField(default=dict)

    unknown_transaction_ids = models.JSONField(default=list)
    review_queue = models.JSONField(default=list)
    warnings = models.JSONField(default=list)

    # Model / processing information
    model = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        blank=True,
        default="",
    )
    prompt_version = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        blank=True,
        default="",
    )

    # Optional large per-transaction tree
    transactions = models.JSONField(
        null=True,
        blank=True,
    )
