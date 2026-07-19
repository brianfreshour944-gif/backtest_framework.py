#!/usr/bin/env python3
"""
Universal Backtesting Framework for Algo Trading Bots
Author: Brian Freshour
Purpose: Reusable, modular backtester that can adapt to ANY bot's strategy logic.
         1. Fetches 1-minute historical data from Alpaca.
         2. Resamples to 5-minute (or custom) candles.
         3. Runs a minute-by-minute simulation.
         4. Tracks positions, PnL, drawdown, and trade logs.
         5. Supports custom Buy/Sell logic via inheritance.

How to Use for a NEW bot:
1. Create a new class that inherits from `Strategy`.
2. Override the `should_buy()` and `should_sell()` methods.
3. Override `get_stop_loss()` and `get_take_profit()` if your bot uses dynamic exits.
4. Run `python backtest_framework.py --config your_config.yaml`

OR just import the engine and run it programmatically.
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import sys
import os

# ---------- CONFIGURATION ----------
@dataclass
class BacktestConfig:
    """Configuration object for the backtest."""
    # Data parameters
    assets: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    start_date: str = "2026-07-18 13:50:00"
    end_date: str = "2026-07-18 18:50:00"
    base_currency: str = "USD"
    timeframe: str = "5Min"  # resample interval
    warmup_bars: int = 256   # minimum bars before first trade

    # Strategy parameters (override with your bot's config)
    buy_threshold: float = 0.51
    sell_threshold: float = 0.45
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.02
    max_hold_hours: float = 4.0
    min_hold_hours: float = 0.5
    max_positions: int = 10
    portfolio_cap: float = 1000.0

    # Alpaca credentials (set env vars ALPACA_API_KEY, ALPACA_SECRET_KEY)
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    paper: bool = True

# ---------- DATA FETCHER ----------
class AlpacaDataFetcher:
    """Fetch historical 1-minute data from Alpaca."""
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://data.alpaca.markets/v2" if not paper else "https://data.alpaca.markets/v2"

    def fetch_minute_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch 1-minute OHLCV data from Alpaca."""
        url = f"{self.base_url}/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start,
            "end": end,
            "limit": 10000,
            "adjustment": "raw"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Alpaca API error: {response.text}")
        data = response.json()
        if symbol not in data.get("bars", {}):
            return pd.DataFrame()
        
        bars = data["bars"][symbol]
        df = pd.DataFrame(bars)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['t'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        # Ensure VWAP exists (Alpaca provides it)
        if 'vwap' not in df.columns:
            # Calculate VWAP if missing: (high + low + close) / 3 * volume / volume_sum (approximate)
            # Better: use Alpaca's provided VWAP, but if absent, we create a synthetic one.
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3  # fallback
        return df

# ---------- DATA PIPELINE (Resampler) ----------
class DataPipeline:
    """Resample 1-minute data to 5-minute (or custom) candles."""
    @staticmethod
    def resample_to_timeframe(df: pd.DataFrame, timeframe: str = "5Min") -> pd.DataFrame:
        """Resample 1-minute OHLCV to the given timeframe."""
        if df.empty:
            return df
        
        # Resample
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        # Calculate VWAP for the resampled period
        # We need to compute volume-weighted average price from the 1-min data
        # Since we have 1-min VWAPs, we can aggregate them correctly:
        # VWAP = sum(volume * vwap) / sum(volume)
        # But if we have 1-min bars with their own vwap, we can do this:
        # Better: use the raw 1-min data to compute a proper VWAP for the 5-min period.
        
        # Using the 1-min VWAP column to compute 5-min VWAP:
        # For each 5-min group, we compute weighted average of 1-min vwaps
        def compute_vwap(group):
            if group['volume'].sum() == 0:
                return np.nan
            return (group['volume'] * group['vwap']).sum() / group['volume'].sum()
        
        vwap_series = df['vwap'].resample(timeframe).apply(lambda g: (g * df.loc[g.index, 'volume']).sum() / df.loc[g.index, 'volume'].sum() if df.loc[g.index, 'volume'].sum() > 0 else np.nan)
        # Or simpler: since we already have 1-min vwap, we can use that
        resampled['vwap'] = vwap_series
        
        # Drop NaN rows
        resampled.dropna(inplace=True)
        return resampled

# ---------- STRATEGY ABSTRACT CLASS ----------
class Strategy(ABC):
    """Abstract Base Class for any trading strategy."""
    def __init__(self, config: BacktestConfig):
        self.config = config

    @abstractmethod
    def should_buy(self, asset: str, signal: float, price: float, trend: str, position_count: int) -> bool:
        """Return True if we should open a new long position."""
        pass

    @abstractmethod
    def should_sell(self, asset: str, signal: float, price: float, entry_price: float, hold_time_hours: float) -> bool:
        """Return True if we should close an existing position (signal reversal or stop logic)."""
        pass

    @abstractmethod
    def get_stop_loss(self, asset: str, entry_price: float, current_price: float) -> float:
        """Return the stop-loss price."""
        pass

    @abstractmethod
    def get_take_profit(self, asset: str, entry_price: float, current_price: float) -> float:
        """Return the take-profit price."""
        pass

# ---------- CONCRETE STRATEGY: Grok Apex Ironclad Bot ----------
class GrokApexStrategy(Strategy):
    """Reproduces the exact logic of the Grok Apex Ironclad Bot v9."""
    
    def should_buy(self, asset: str, signal: float, price: float, trend: str, position_count: int) -> bool:
        # 1. Check max positions
        if position_count >= self.config.max_positions:
            return False
        # 2. Check signal threshold
        if signal < self.config.buy_threshold:
            return False
        # 3. Check trend filter (optional, but we used it)
        if trend != "up":
            return False
        return True

    def should_sell(self, asset: str, signal: float, price: float, entry_price: float, hold_time_hours: float) -> bool:
        # 1. Signal reversal
        if signal < self.config.sell_threshold:
            return True
        # 2. Max hold time
        if hold_time_hours >= self.config.max_hold_hours:
            return True
        return False

    def get_stop_loss(self, asset: str, entry_price: float, current_price: float) -> float:
        # Fixed percentage stop loss
        return entry_price * (1 - self.config.stop_loss_pct)

    def get_take_profit(self, asset: str, entry_price: float, current_price: float) -> float:
        # Fixed percentage take profit
        return entry_price * (1 + self.config.take_profit_pct)

# ---------- PLACEHOLDER: Apex Oracle Strategy (for future use) ----------
class ApexOracleStrategy(Strategy):
    """
    Placeholder for the Apex Oracle bot.
    TODO: Once you reverse-engineer its logic, override these methods.
    """
    def should_buy(self, asset: str, signal: float, price: float, trend: str, position_count: int) -> bool:
        # Example: if signal > 0.55 and trend == 'up'
        # Replace with actual Apex Oracle logic
        return False

    def should_sell(self, asset: str, signal: float, price: float, entry_price: float, hold_time_hours: float) -> bool:
        # Example: if signal < 0.45 or hold_time > 4h
        return False

    def get_stop_loss(self, asset: str, entry_price: float, current_price: float) -> float:
        return entry_price * 0.97  # 3% stop

    def get_take_profit(self, asset: str, entry_price: float, current_price: float) -> float:
        return entry_price * 1.02  # 2% profit

# ---------- BACKTEST ENGINE ----------
class BacktestEngine:
    """Simulates trades minute-by-minute."""
    def __init__(self, config: BacktestConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger("BacktestEngine")
        self.positions = {}  # asset -> dict with entry_price, entry_time, signal
        self.equity_curve = []
        self.trades = []
        self.cash = config.portfolio_cap
        self.total_value = config.portfolio_cap
        self.signal_cache = {}  # asset -> latest signal

    def run(self) -> Dict[str, Any]:
        """Main simulation loop."""
        self.logger.info("🚀 Starting backtest...")
        fetcher = AlpacaDataFetcher(
            api_key=self.config.alpaca_api_key or os.getenv("ALPACA_API_KEY"),
            secret_key=self.config.alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY"),
            paper=self.config.paper
        )

        # 1. Fetch all data for all assets
        all_data = {}
        for asset in self.config.assets:
            self.logger.info(f"📥 Fetching data for {asset}...")
            df = fetcher.fetch_minute_bars(asset, self.config.start_date, self.config.end_date)
            if df.empty:
                self.logger.warning(f"⚠️ No data for {asset}, skipping.")
                continue
            # Resample to configured timeframe
            df_resampled = DataPipeline.resample_to_timeframe(df, self.config.timeframe)
            all_data[asset] = df_resampled

        # 2. Get the union of all timestamps (minute-by-minute simulation)
        # We will iterate over each minute from start to end.
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        current_time = start_dt

        # Pre-compute signals for each asset at each timestamp?
        # Instead, we will simulate minute-by-minute and fetch the current bar's data.

        self.logger.info(f"🕒 Simulating from {start_dt} to {end_dt}")
        
        # Prepare data slices: we need to simulate a "live" feed where data arrives minute by minute.
        # Since we have all data upfront, we slice up to the current time.
        minute_index = pd.date_range(start=start_dt, end=end_dt, freq='1Min')
        
        for sim_time in minute_index:
            # For each asset, get the latest 5-min bar that is fully closed.
            # We simulate the closed-candle gate: only process if sim_time.second >= 2 and minute % 5 == 0
            # But since we are using minute-by-minute, we check if the current minute is a boundary.
            # We only execute on the first minute of each new 5-minute candle.
            if sim_time.minute % 5 != 0:
                # Skip unless we are exactly at the start of a new candle
                # We can still update PnL and check stops, but no new signals
                pass

            # Check for stop-loss and take-profit on existing positions
            self._update_positions(sim_time, all_data)

            # Execute new buy signals (only at candle boundaries, e.g., sim_time.second >= 2)
            # In minute simulation, we check if we are at the first minute of a 5-min block
            if sim_time.minute % 5 == 0:
                # We are at a potential boundary (0,5,10,...)
                # Fetch the signal for this asset at this exact time.
                for asset, df in all_data.items():
                    # Get the bar that closes at sim_time (or the latest bar up to sim_time)
                    # Since we have resampled to 5-min, we find the bar that ends at sim_time
                    if sim_time in df.index:
                        row = df.loc[sim_time]
                        # Compute trend and signal (this is where ML prediction would go)
                        # For the universal framework, we need to call a prediction function.
                        # We'll use a placeholder: signal = row['close'] / sma? 
                        # BUT for the Grok bot, we need the ML signal.
                        # So we must pass a signal generator.
                        # For now, let's assume we have a function that returns signal and trend.
                        # We will implement a placeholder signal generator.
                        signal, trend = self._get_signal(asset, row, sim_time)
                        self.signal_cache[asset] = signal
                        # Check if we should buy
                        pos_count = len(self.positions)
                        if self.strategy.should_buy(asset, signal, row['close'], trend, pos_count):
                            self._open_position(asset, row['close'], sim_time, signal)
                    else:
                        # If no bar at this exact time, use the latest available
                        pass

            # Update equity curve
            self._update_equity(sim_time, all_data)

        # Close all positions at the end
        for asset in list(self.positions.keys()):
            self._close_position(asset, all_data[asset].iloc[-1]['close'], sim_time, reason="End of simulation")

        # Compute metrics
        return self._compute_metrics()

    def _get_signal(self, asset: str, row: pd.Series, sim_time: pd.Timestamp) -> Tuple[float, str]:
        """
        Placeholder for ML signal generation.
        For the Grok bot, we need to load the model and scaler.
        Since this is a universal framework, we will inject a signal provider.
        """
        # For demonstration, we just return a fake signal.
        # In practice, you would pass a pre-computed signal dataframe.
        # To truly replicate, we would need to run the model here.
        # Let's create a fallback: look for a precomputed signal dict.
        if hasattr(self, 'signal_data') and asset in self.signal_data:
            # If we precomputed signals for each timestamp
            return self.signal_data[asset].get(sim_time, (0.5, 'neutral'))
        
        # Fallback: Synthetic signal based on price movement relative to SMA
        # THIS IS A PLACEHOLDER. REPLACE WITH YOUR ML PREDICTOR.
        return 0.50, "up"

    def _update_positions(self, sim_time: pd.Timestamp, all_data: Dict[str, pd.DataFrame]):
        """Check if any positions should be closed due to SL/TP or signal reversal."""
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            # Get current price
            df = all_data.get(asset)
            if df is None or df.empty:
                continue
            # Get the latest price up to sim_time
            current_prices = df[df.index <= sim_time]
            if current_prices.empty:
                continue
            current_price = current_prices.iloc[-1]['close']
            
            # Check SL / TP
            sl = self.strategy.get_stop_loss(asset, pos['entry_price'], current_price)
            tp = self.strategy.get_take_profit(asset, pos['entry_price'], current_price)
            
            if current_price <= sl:
                self._close_position(asset, current_price, sim_time, reason="Stop Loss")
                continue
            if current_price >= tp:
                self._close_position(asset, current_price, sim_time, reason="Take Profit")
                continue
            
            # Check hold time
            hold_time = (sim_time - pos['entry_time']).total_seconds() / 3600.0
            if hold_time >= self.config.max_hold_hours:
                self._close_position(asset, current_price, sim_time, reason="Max Hold Time")
                continue
            
            # Check signal reversal (if we have a cached signal)
            if asset in self.signal_cache:
                signal = self.signal_cache[asset]
                if self.strategy.should_sell(asset, signal, current_price, pos['entry_price'], hold_time):
                    self._close_position(asset, current_price, sim_time, reason="Signal Reversal")

    def _open_position(self, asset: str, price: float, time: pd.Timestamp, signal: float):
        """Open a new position."""
        # Calculate quantity based on portfolio cap / max_positions
        qty = (self.config.portfolio_cap / self.config.max_positions) / price
        self.positions[asset] = {
            'entry_price': price,
            'entry_time': time,
            'signal': signal,
            'qty': qty,
            'status': 'open'
        }
        self.cash -= qty * price
        self.logger.info(f"🟢 BUY {asset} @ {price:.2f} | Signal: {signal:.4f} | Time: {time}")

    def _close_position(self, asset: str, price: float, time: pd.Timestamp, reason: str):
        """Close an existing position."""
        if asset not in self.positions:
            return
        pos = self.positions[asset]
        qty = pos['qty']
        self.cash += qty * price
        pnl = (price - pos['entry_price']) * qty
        pnl_pct = (price / pos['entry_price'] - 1) * 100
        self.trades.append({
            'asset': asset,
            'entry_time': pos['entry_time'],
            'exit_time': time,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_hours': (time - pos['entry_time']).total_seconds() / 3600.0
        })
        self.logger.info(f"🔴 SELL {asset} @ {price:.2f} | PnL: {pnl_pct:+.2f}% | Reason: {reason}")
        del self.positions[asset]

    def _update_equity(self, sim_time: pd.Timestamp, all_data: Dict[str, pd.DataFrame]):
        """Update the equity curve."""
        total_market_value = 0
        for asset, pos in self.positions.items():
            df = all_data.get(asset)
            if df is None or df.empty:
                continue
            current_prices = df[df.index <= sim_time]
            if current_prices.empty:
                continue
            price = current_prices.iloc[-1]['close']
            total_market_value += pos['qty'] * price
        
        total_equity = self.cash + total_market_value
        drawdown = (total_equity / self.config.portfolio_cap - 1) * 100 if self.config.portfolio_cap > 0 else 0
        self.equity_curve.append({
            'timestamp': sim_time,
            'cash': self.cash,
            'market_value': total_market_value,
            'equity': total_equity,
            'drawdown_pct': drawdown
        })

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute final performance metrics."""
        if not self.equity_curve:
            return {"total_return": 0.0, "trades": 0, "win_rate": 0.0, "max_drawdown": 0.0}
        
        df_equity = pd.DataFrame(self.equity_curve)
        final_equity = df_equity.iloc[-1]['equity']
        initial_equity = df_equity.iloc[0]['equity']
        total_return = (final_equity / initial_equity - 1) * 100
        
        max_drawdown = df_equity['drawdown_pct'].min()
        
        if not self.trades:
            win_rate = 0
        else:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trades) * 100
        
        return {
            "total_return_pct": total_return,
            "total_trades": len(self.trades),
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_drawdown,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }

# ---------- MAIN ENTRY POINT ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger("Main")

    # 1. Define your configuration
    config = BacktestConfig(
        assets=["BTC/USD", "ETH/USD", "AAVE/USD", "LTC/USD", "GRT/USD", "BAT/USD"],
        start_date="2026-07-18 13:50:00",
        end_date="2026-07-18 18:50:00",
        buy_threshold=0.51,
        sell_threshold=0.45,
        stop_loss_pct=0.03,
        take_profit_pct=0.02,
        max_hold_hours=4.0,
        min_hold_hours=0.5,
        max_positions=10,
        portfolio_cap=1000.0,
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True
    )

    # 2. Choose your strategy
    # For Grok Apex:
    strategy = GrokApexStrategy(config)
    
    # For Apex Oracle (when ready):
    # strategy = ApexOracleStrategy(config)

    # 3. Run the backtest
    engine = BacktestEngine(config, strategy)
    results = engine.run()

    # 4. Print results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print("\nTrade Log:")
    for trade in results['trades']:
        print(f"  {trade['asset']} | Entry: {trade['entry_price']:.2f} | Exit: {trade['exit_price']:.2f} | PnL: {trade['pnl_pct']:+.2f}% | Reason: {trade['reason']}")
    print("="*60)
