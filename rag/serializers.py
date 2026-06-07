from rest_framework import serializers
from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'chunk_count', 'uploaded_at']
        read_only_fields = ['id', 'chunk_count', 'uploaded_at']


class DocumentUploadSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    file = serializers.FileField()

    def validate_file(self, value):
        if not value.name.endswith('.pdf'):
            raise serializers.ValidationError("Only PDF files are supported.")
        if value.size > 10 * 1024 * 1024:  # 10MB limit
            raise serializers.ValidationError("File size cannot exceed 10MB.")
        return value


class QuerySerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)
    document_id = serializers.UUIDField()