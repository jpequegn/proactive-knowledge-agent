"""Tests for embedding service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from openai import RateLimitError

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

    @pytest.fixture
    def mock_cache(self):
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

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
    async def test_generate_cached(self, mock_client, mock_cache):
        """Test generating embedding with cache hit."""
        service = EmbeddingService(api_key="test", cache=mock_cache)
        
        # Setup cache hit
        mock_cache.get.return_value = json.dumps([0.9, 0.9])
        
        emb = await service.generate("cached text")
        
        assert emb == [0.9, 0.9]
        mock_cache.get.assert_called_once()
        # API should NOT be called
        mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_cache_miss(self, mock_client, mock_cache):
        """Test generating embedding with cache miss."""
        service = EmbeddingService(api_key="test", cache=mock_cache)
        
        # Setup cache miss
        mock_cache.get.return_value = None
        
        # Setup API response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5, 0.5])]
        mock_client.embeddings.create.return_value = mock_response
        
        emb = await service.generate("new text")
        
        assert emb == [0.5, 0.5]
        mock_cache.get.assert_called_once()
        mock_client.embeddings.create.assert_called_once()
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_batch_mixed_cache(self, mock_client, mock_cache):
        """Test batch generation with some items cached."""
        service = EmbeddingService(api_key="test", batch_size=2, cache=mock_cache)
        
        texts = ["cached1", "new1", "cached2", "new2"]
        
        # Mock cache responses
        async def mock_get(key):
            if "cached1" in key: # Relying on hash containing partial key is wrong, but keys are hashed.
                # We need to know the hash.
                pass 
            return None
            
        # Better approach: verify keys match expected hashes, or just side_effect based on call order/args?
        # Since we can't predict hash easily in test without replicating logic, 
        # we can just assume order or mock the _generate_cache_key method?
        # Or simply trust the flow.
        
        # Let's patch _generate_cache_key to return predictable keys
        with patch.object(service, '_generate_cache_key', side_effect=lambda x: f"key:{x}"):
            mock_cache.get.side_effect = [
                json.dumps([0.1]), # cached1
                None,              # new1
                json.dumps([0.3]), # cached2
                None               # new2
            ]
            
            # Mock API for the 2 new items
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.2]), # new1
                MagicMock(embedding=[0.4])  # new2
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            embeddings = await service.generate_batch(texts)
            
            assert embeddings[0] == [0.1]
            assert embeddings[1] == [0.2]
            assert embeddings[2] == [0.3]
            assert embeddings[3] == [0.4]
            
            # Should have called API once with the batch of 2 new items
            mock_client.embeddings.create.assert_called_once()
            call_args = mock_client.embeddings.create.call_args
            assert call_args.kwargs['input'] == ["new1", "new2"]

    @pytest.mark.asyncio
    async def test_retry_logic(self, mock_client):
        """Test retry logic on RateLimitError."""
        service = EmbeddingService(api_key="test")
        
        # Mock API to fail twice with RateLimitError, then succeed
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        
        mock_client.embeddings.create.side_effect = [
            RateLimitError(message="Rate limit", response=MagicMock(), body={}),
            RateLimitError(message="Rate limit", response=MagicMock(), body={}),
            mock_response
        ]
        
        # Patch sleep to avoid actual waiting
        with patch("asyncio.sleep", new_callable=AsyncMock):
            emb = await service.generate("text")
            
        assert emb == [0.1]
        assert mock_client.embeddings.create.call_count == 3