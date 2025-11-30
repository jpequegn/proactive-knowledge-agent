"""Market data client using yfinance."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog
import yfinance as yf

from src.models import MarketOHLCV, MarketSyncResult

logger = structlog.get_logger()


class MarketClient:
    """Client for fetching market data."""

    async def get_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> list[MarketOHLCV]:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            period: Data period to download
            interval: Data interval
        """
        try:
            # Run blocking yfinance call in thread pool
            return await asyncio.to_thread(
                self._fetch_sync, symbol, period, interval
            )

        except Exception as e:
            logger.error(
                "Failed to fetch market data",
                symbol=symbol,
                error=str(e)
            )
            raise

    def _fetch_sync(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> list[MarketOHLCV]:
        """Synchronous fetch implementation."""
        ticker = yf.Ticker(symbol)
        # auto_adjust=True is default, so Close is Adjusted Close
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning("No data found for symbol", symbol=symbol)
            return []

        results = []
        for index, row in df.iterrows():
            # index is Timestamp (datetime64)
            date = index.to_pydatetime()
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)

            results.append(
                MarketOHLCV(
                    symbol=symbol,
                    date=date,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adjusted_close=float(row["Close"]),
                )
            )
        return results

    def calculate_sma(
        self,
        data: list[MarketOHLCV],
        window: int = 20
    ) -> list[float]:
        """Calculate Simple Moving Average."""
        if not data:
            return []

        closes = pd.Series([d.close for d in data])
        sma = closes.rolling(window=window).mean()
        return sma.fillna(0).tolist()

    def calculate_rsi(
        self,
        data: list[MarketOHLCV],
        window: int = 14
    ) -> list[float]:
        """Calculate Relative Strength Index."""
        if not data:
            return []

        closes = pd.Series([d.close for d in data])
        delta = closes.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0).tolist()


async def sync_market_data(
    symbols: list[str],
    period: str = "1mo",
) -> MarketSyncResult:
    """Sync market data for list of symbols."""
    result = MarketSyncResult(started_at=datetime.now(UTC))
    client = MarketClient()

    for symbol in symbols:
        try:
            data = await client.get_history(symbol, period=period)
            result.data.extend(data)
            result.data_points_fetched += len(data)
            result.symbols_processed += 1

            logger.info(
                "Market data synced",
                symbol=symbol,
                points=len(data),
            )

        except Exception as e:
            result.errors.append(f"{symbol}: {str(e)}")
            result.success = False
            # Continue with other symbols

    result.completed_at = datetime.now(UTC)
    return result
