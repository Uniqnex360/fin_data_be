from django.urls import path

from apps.bank.views import (
    BankFilesUploadAPIView,
    BankDocsAPIView,
    ClusterBankStatementAPIView,
    ClusterBankStatementAPIViewV2,
)

API_URL_PREFIX = "api/bank"

urlpatterns = [
    path(
        f"{API_URL_PREFIX}/file/upload/",
        BankFilesUploadAPIView.as_view(),
    ),
    path(
        f"{API_URL_PREFIX}/docs/",
        BankDocsAPIView.as_view(),
    ),
    path(
        f"{API_URL_PREFIX}/cluster/docs/",
        ClusterBankStatementAPIView.as_view(),
    ),
    path(
        f"{API_URL_PREFIX}/v2/cluster/docs/",
        ClusterBankStatementAPIViewV2.as_view(),
    ),
]
