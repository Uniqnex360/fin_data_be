from django.urls import path
from apps.common.router import AppSimpleRouter

from apps.bank.views import (
    BankFilesUploadAPIView,
    BankDocsAPIView,
    ClusterBankStatementAPIView,
    ClusterBankStatementAPIViewV2,
    BankFileListAPIViewSet,
    BankStatementDetailAPIView,
)

API_URL_PREFIX = "api/bank"

router = AppSimpleRouter()


router.register(f"{API_URL_PREFIX}/file/list", BankFileListAPIViewSet)

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
    path(
        f"{API_URL_PREFIX}/statement/detail/<int:id>",
        BankStatementDetailAPIView.as_view(),
    ),
] + router.urls
