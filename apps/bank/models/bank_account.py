from django.db import models
from apps.common.models import BaseFileUploadModel, UploadToFolder, BaseModel, COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG, COMMON_CHAR_FIELD_MAX_LENGTH


class BankDocumentFile(BaseFileUploadModel):
    file = models.FileField(upload_to=UploadToFolder("bank_document_file"), max_length=500)


# bank statement cluster data
class BankAccountBase(BaseModel):

    file = models.ForeignKey(BankDocumentFile, on_delete=models.CASCADE)
    bank_name = models.CharField(max_length=COMMON_CHAR_FIELD_MAX_LENGTH, **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG
)

    account_holder_name = models.CharField(max_length=COMMON_CHAR_FIELD_MAX_LENGTH, **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG
                                 )
    account_number=models.CharField(max_length=COMMON_CHAR_FIELD_MAX_LENGTH, **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG
                            ),
    ifsc=models.CharField(max_length=COMMON_CHAR_FIELD_MAX_LENGTH, **COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG
                            ),
    statement_period_start= models.DateField(**COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG),
    "statement_period_end"= models.DateField(**COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG) ,
    "opening_balance"= models.FloatField(**COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG),
    "closing_balance"=models.FloatField(**COMMON_BLANK_AND_NULLABLE_FIELD_CONFIG)