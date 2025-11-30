"""Tests for Market Data Client."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.market_client import MarketClient, sync_market_data
from src.models import MarketOHLCV


class TestMarketClient:
    """Tests for MarketClient."""

    @pytest.fixture
    def mock_ticker(self):
        with patch("yfinance.Ticker") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_get_history_success(self, mock_ticker):
        """Test fetching history successfully."""
        # Setup mock data
        dates = pd.date_range(start="2024-01-01", periods=2)
        data = {
            "Open": [150.0, 152.0],
            "High": [155.0, 156.0],
            "Low": [149.0, 151.0],
            "Close": [152.0, 155.0],
            "Volume": [1000000, 1200000],
        }
        df = pd.DataFrame(data, index=dates)

        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_ticker.return_value = mock_instance

        client = MarketClient()
        result = await client.get_history("AAPL")

        assert len(result) == 2
        assert result[0].symbol == "AAPL"
        assert result[0].open == 150.0
        assert result[0].close == 152.0
        assert result[1].date.year == 2024

    @pytest.mark.asyncio
    async def test_get_history_empty(self, mock_ticker):
        """Test fetching history with no data."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        client = MarketClient()
        result = await client.get_history("INVALID")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_history_failure(self, mock_ticker):
        """Test fetching history failure."""
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_instance

        client = MarketClient()
        with pytest.raises(Exception, match="API Error"):
            await client.get_history("FAIL")

    def test_calculate_sma(self):
        """Test SMA calculation."""
        client = MarketClient()
        
        # Test empty
        assert client.calculate_sma([]) == []

        data = []
        base_date = datetime.now(UTC)
        for i in range(5):
            data.append(
                MarketOHLCV(
                    symbol="TEST",
                    date=base_date,
                    open=100, high=110, low=90,
                    close=100 + i * 10,  # 100, 110, 120, 130, 140
                    volume=100
                )
            )

        sma = client.calculate_sma(data, window=2)
        # expected:
        # 0: NaN -> 0
        # 1: (100+110)/2 = 105
        # 2: (110+120)/2 = 115
        # 3: (120+130)/2 = 125
        # 4: (130+140)/2 = 135

        assert len(sma) == 5
        assert sma[1] == 105.0
        assert sma[4] == 135.0

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        client = MarketClient()

        # Test empty
        assert client.calculate_rsi([]) == []

        data = []
        base_date = datetime.now(UTC)

        # Create a sequence: Up, Up, Down
        prices = [100, 110, 120, 110]
        for p in prices:
            data.append(
                MarketOHLCV(
                    symbol="TEST",
                    date=base_date,
                    open=100, high=110, low=90,
                    close=p,
                    volume=100
                )
            )

        # window=2
        # diff: [NaN, 10, 10, -10]
        # gain: [NaN, 10, 10, 0] -> avg gain (window=2) -> [NaN, NaN, 10, 5]
        # loss: [NaN, 0, 0, 10] -> avg loss (window=2) -> [NaN, NaN, 0, 5]
        # rs: [NaN, NaN, inf, 1]
        # rsi: [NaN, NaN, 100, 50] -> fillna(0) -> [0, 0, 100, 50]

        rsi = client.calculate_rsi(data, window=2)

        assert len(rsi) == 4
        assert rsi[2] == 100.0
        assert rsi[3] == 50.0


@pytest.mark.asyncio
async def test_sync_market_data():
    """Test full sync orchestration."""
    with patch("src.ingestion.market_client.MarketClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_history = MagicMock()

        # Setup async return value
        async def mock_get_history(*args, **kwargs):
            return [
                MarketOHLCV(
                    symbol="AAPL",
                    date=datetime.now(UTC),
                    open=100, high=110, low=90, close=105, volume=1000
                )
            ]
        mock_instance.get_history.side_effect = mock_get_history

        result = await sync_market_data(["AAPL"])

        assert result.success is True
        assert result.symbols_processed == 1
        assert len(result.data) == 1


@pytest.mark.asyncio
async def test_sync_market_data_partial_failure():
    """Test sync with some failures."""
    with patch("src.ingestion.market_client.MarketClient") as MockClient:
        mock_instance = MockClient.return_value

        async def mock_get_history(symbol, **kwargs):
            if symbol == "FAIL":
                raise Exception("Sync Error")
            return []

        mock_instance.get_history.side_effect = mock_get_history

        result = await sync_market_data(["AAPL", "FAIL"])

        assert result.success is False
        assert len(result.errors) == 1
        assert "FAIL: Sync Error" in result.errors[0]
        # AAPL succeeded (returned empty list) so it counts as processed
        assert result.symbols_processed == 1
