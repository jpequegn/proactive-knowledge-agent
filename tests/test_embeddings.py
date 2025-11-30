"""Tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.embeddings import EmbeddingService


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_client(self):
        with patch("src.ingestion.embeddings.AsyncOpenAI") as MockClient:
            mock_instance = MockClient.return_value
            # Configure embeddings.create to be awaitable
            mock_instance.embeddings.create = AsyncMock()
            yield mock_instance

    @pytest.mark.asyncio
    async def test_generate_single(self, mock_client):
            """Test generating single embedding."""

            service = EmbeddingService(api_key="test")

            

            mock_response = MagicMock()

            mock_response.data = [MagicMock(embedding=[0.1, 0.2])]

            mock_client.embeddings.create.return_value = mock_response

    

            emb = await service.generate("test text")

            

            assert emb == [0.1, 0.2]

            mock_client.embeddings.create.assert_called_once()

    

    @pytest.mark.asyncio
    async def test_generate_single_empty(self, mock_client):
        """Test generating embedding for empty text."""
        service = EmbeddingService(api_key="test")
        emb = await service.generate("")
        assert len(emb) == 1536
        assert all(x == 0.0 for x in emb)

    @pytest.mark.asyncio
    async def test_generate_batch(self, mock_client):
        """Test generating batch embeddings."""
        service = EmbeddingService(api_key="test", batch_size=2)
        
        texts = ["text1", "text2", "text3"]
        
        # Mock response for first batch (text1, text2)
        mock_resp1 = MagicMock()
        mock_resp1.data = [
            MagicMock(embedding=[0.1]),
            MagicMock(embedding=[0.2])
        ]
        
        # Mock response for second batch (text3)
        mock_resp2 = MagicMock()
        mock_resp2.data = [
            MagicMock(embedding=[0.3])
        ]
        
        mock_client.embeddings.create.side_effect = [mock_resp1, mock_resp2]

        embeddings = await service.generate_batch(texts)
        
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1]
        assert embeddings[1] == [0.2]
        assert embeddings[2] == [0.3]
        
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, mock_client):
        """Test cosine similarity calculation."""
        service = EmbeddingService(api_key="test")
        
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        sim = await service.cosine_similarity(v1, v2)
        assert abs(sim - 1.0) < 0.0001
        
        v3 = [0.0, 1.0]
        sim = await service.cosine_similarity(v1, v3)
        assert abs(sim - 0.0) < 0.0001

    

        @pytest.mark.asyncio

        async def test_generate_batch(self, mock_client):

            """Test generating batch embeddings."""

            service = EmbeddingService(api_key="test", batch_size=2)

            

            texts = ["text1", "text2", "text3"]

            

            # Mock response for first batch (text1, text2)

            mock_resp1 = MagicMock()

            mock_resp1.data = [

                MagicMock(embedding=[0.1]),

                MagicMock(embedding=[0.2])

            ]

            

            # Mock response for second batch (text3)

            mock_resp2 = MagicMock()

            mock_resp2.data = [

                MagicMock(embedding=[0.3])

            ]

            

            mock_client.embeddings.create.side_effect = [mock_resp1, mock_resp2]

    

            embeddings = await service.generate_batch(texts)

            

            assert len(embeddings) == 3

            assert embeddings[0] == [0.1]

            assert embeddings[1] == [0.2]

            assert embeddings[2] == [0.3]

            

            assert mock_client.embeddings.create.call_count == 2

    

        @pytest.mark.asyncio

        async def test_cosine_similarity(self, mock_client):

            """Test cosine similarity calculation."""

            service = EmbeddingService(api_key="test")

            

            v1 = [1.0, 0.0]

            v2 = [1.0, 0.0]

            sim = await service.cosine_similarity(v1, v2)

            assert abs(sim - 1.0) < 0.0001

            

            v3 = [0.0, 1.0]

            sim = await service.cosine_similarity(v1, v3)

            assert abs(sim - 0.0) < 0.0001

    