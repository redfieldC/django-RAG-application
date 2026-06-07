from django.urls import path
from .views import DocumentListView, DocumentUploadView, QueryView

urlpatterns = [
    path('documents/', DocumentListView.as_view(), name='document-list'),
    path('documents/upload/', DocumentUploadView.as_view(), name='document-upload'),
    path('query/', QueryView.as_view(), name='query'),
]