from django.urls import path
from apps.common.router import AppSimpleRouter

from apps.bank.views import (
    BankFilesUploadAPIView,
    BankDocsAPIView,
    ClusterBankStatementAPIView,
    ClusterBankStatementAPIViewV2,
    BankFileListAPIViewSet,
    BankStatementDetailAPIView,
    ManualDataExtractionView,
    ManualBankStatementDetailAPIView,
    ManualExtractionListAPIViewSet,
)

API_URL_PREFIX = "api/bank"

router = AppSimpleRouter()


router.register(f"{API_URL_PREFIX}/file/list", BankFileListAPIViewSet)
router.register(f"{API_URL_PREFIX}/manual/list", ManualExtractionListAPIViewSet)

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
    # manual
    path(
        f"{API_URL_PREFIX}/statement/manual-extraction/",
        ManualDataExtractionView.as_view(),
    ),
    path(
        f"{API_URL_PREFIX}/manual/detail/<int:id>",
        ManualBankStatementDetailAPIView.as_view(),
    ),
] + router.urls
