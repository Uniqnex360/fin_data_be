from apps.common.serializers import AppReadOnlyModelSerializer
from apps.bank.models import ManualBankAccountBase

class ManualBankAccountBaseSerializer(AppReadOnlyModelSerializer):
    class Meta(AppReadOnlyModelSerializer.Meta):
        model = ManualBankAccountBase
        fields = ["id", "business_name", "created"]