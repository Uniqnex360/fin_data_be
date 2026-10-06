from apps.common.serializers import AppReadOnlyModelSerializer
from apps.bank.models import BankDocumentFile


class BankDocumentFileSerializer(AppReadOnlyModelSerializer):
    class Meta(AppReadOnlyModelSerializer.Meta):
        model = BankDocumentFile
        fields = ["id", "file", "created"]