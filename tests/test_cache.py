"""Tests for Redis cache wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache import Cache


class TestCache:
    """Tests for Cache class."""

    @pytest.fixture
    def mock_redis(self):
        with patch("src.cache.Redis") as mock:
            mock_instance = AsyncMock()
            mock.from_url.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_connect(self, mock_redis):
        """Test connection."""
        cache = Cache("redis://localhost")
        await cache.connect()
        
        assert cache.redis is not None
        cache.redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_get(self, mock_redis):
        """Test set and get."""
        cache = Cache("redis://localhost")
        await cache.connect()
        
        await cache.set("key", "value")
        cache.redis.set.assert_called_with("key", "value", ex=None)
        
        mock_redis.get.return_value = "value"
        val = await cache.get("key")
        assert val == "value"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        """Test error when not connected."""
        cache = Cache("redis://localhost")
        with pytest.raises(RuntimeError):
            await cache.get("key")
