from django.db import models

from apps.common.models import BaseModel, COMMON_CHAR_FIELD_MAX_LENGTH, COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG


class ManualBankAccountBase(BaseModel):

    # Account summary
    business_name = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    account_reference = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    business_type = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
        **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG,
    )
    statement_period = models.CharField(
        max_length=COMMON_CHAR_FIELD_MAX_LENGTH,
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

    # Raw transaction data
    transactions = models.JSONField(
        default=list,
    )

    # Transaction clustering / analytical results
    clusters = models.JSONField(
        default=list,
    )

    # Optional model / processing information
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
