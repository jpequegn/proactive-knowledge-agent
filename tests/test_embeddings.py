"""Tests for embedding generation service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.embeddings import EmbeddingService, EMBEDDING_DIMENSIONS


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create mock OpenAI embedding response."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * EMBEDDING_DIMENSIONS

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        return mock_response

    @pytest.mark.asyncio
    async def test_generate_single_embedding(
        self,
        mock_openai_response: MagicMock,
    ) -> None:
        """Test single embedding generation."""
        with patch("src.ingestion.embeddings.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
            mock_client_class.return_value = mock_client

            service = EmbeddingService(api_key="test-key")
            embedding = await service.generate("Test text")

            assert len(embedding) == EMBEDDING_DIMENSIONS
            mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_empty_text_returns_zeros(self) -> None:
        """Test empty text returns zero vector."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            embedding = await service.generate("")

            assert len(embedding) == EMBEDDING_DIMENSIONS
            assert all(x == 0.0 for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self) -> None:
        """Test batch embedding generation."""
        mock_embedding1 = MagicMock()
        mock_embedding1.embedding = [0.1] * EMBEDDING_DIMENSIONS
        mock_embedding2 = MagicMock()
        mock_embedding2.embedding = [0.2] * EMBEDDING_DIMENSIONS

        mock_response = MagicMock()
        mock_response.data = [mock_embedding1, mock_embedding2]

        with patch("src.ingestion.embeddings.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = EmbeddingService(api_key="test-key")
            embeddings = await service.generate_batch(["Text 1", "Text 2"])

            assert len(embeddings) == 2
            assert len(embeddings[0]) == EMBEDDING_DIMENSIONS
            assert len(embeddings[1]) == EMBEDDING_DIMENSIONS

    @pytest.mark.asyncio
    async def test_generate_batch_with_empty_texts(self) -> None:
        """Test batch generation handles empty texts."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * EMBEDDING_DIMENSIONS

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        with patch("src.ingestion.embeddings.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = EmbeddingService(api_key="test-key")
            embeddings = await service.generate_batch(["Valid text", "", "  "])

            # Only valid text should be processed
            assert len(embeddings) == 3
            # Empty texts get zero vectors
            assert all(x == 0.0 for x in embeddings[1])
            assert all(x == 0.0 for x in embeddings[2])

    @pytest.mark.asyncio
    async def test_generate_batch_empty_list(self) -> None:
        """Test empty batch returns empty list."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            embeddings = await service.generate_batch([])

            assert embeddings == []

    @pytest.mark.asyncio
    async def test_cosine_similarity_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors is 1."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            vec = [1.0, 2.0, 3.0]
            similarity = await service.cosine_similarity(vec, vec)

            assert abs(similarity - 1.0) < 0.0001

    @pytest.mark.asyncio
    async def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors is 0."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            vec1 = [1.0, 0.0]
            vec2 = [0.0, 1.0]
            similarity = await service.cosine_similarity(vec1, vec2)

            assert abs(similarity) < 0.0001

    @pytest.mark.asyncio
    async def test_cosine_similarity_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors is -1."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            vec1 = [1.0, 2.0, 3.0]
            vec2 = [-1.0, -2.0, -3.0]
            similarity = await service.cosine_similarity(vec1, vec2)

            assert abs(similarity + 1.0) < 0.0001

    @pytest.mark.asyncio
    async def test_cosine_similarity_zero_vector(self) -> None:
        """Test cosine similarity with zero vector returns 0."""
        with patch("src.ingestion.embeddings.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")
            vec1 = [1.0, 2.0, 3.0]
            vec2 = [0.0, 0.0, 0.0]
            similarity = await service.cosine_similarity(vec1, vec2)

            assert similarity == 0.0
