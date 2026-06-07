import uuid
import os
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .models import Document
from .serializers import DocumentSerializer, DocumentUploadSerializer, QuerySerializer
from .services import ingest_document, query_document


class DocumentListView(APIView):
    """GET /api/documents/ — List all uploaded documents"""

    def get(self, request):
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)


class DocumentUploadView(APIView):
    """POST /api/documents/upload/ — Upload and process a PDF"""

    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        title = serializer.validated_data['title']
        file = serializer.validated_data['file']

        # Save file to disk
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'documents'), exist_ok=True)

        doc = Document.objects.create(
            title=title,
            file=file,
            collection_name=str(uuid.uuid4()).replace('-', '_')
        )

        try:
            # Run LangChain ingestion pipeline
            file_path = doc.file.path
            chunk_count = ingest_document(file_path, doc.collection_name)
            doc.chunk_count = chunk_count
            doc.save()
        except Exception as e:
            traceback.print_exc()  # add this line
            doc.delete()
            return Response(
                {"error": f"Failed to process document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(
            {
                "message": "Document uploaded and processed successfully.",
                "document": DocumentSerializer(doc).data
            },
            status=status.HTTP_201_CREATED
        )


class QueryView(APIView):
    """POST /api/query/ — Ask a question about a document"""

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        question = serializer.validated_data['question']
        document_id = serializer.validated_data['document_id']

        try:
            doc = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response(
                {"error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            answer = query_document(question, doc.collection_name)
        except Exception as e:
            return Response(
                {"error": f"Query failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            "document": doc.title,
            "question": question,
            "answer": answer
        })