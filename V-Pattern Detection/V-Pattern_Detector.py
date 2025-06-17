import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VPatternDetector:
    """
    V Pattern Detection System
    Detects bullish and bearish V reversal patterns according to specified conditions
    """
    
    def __init__(self):
        self.patterns = []
        self.performance_stats = {}
    
    def detect_v_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Main function to detect V patterns in OHLCV data
        
        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        
        Returns:
            List of detected V patterns with details
        """
        patterns = []
        
        # Need at least 5 candles to detect a pattern (1 for L1 + max 3 for L2 + 1 confirmation)
        for i in range(1, len(df) - 4):
            # Check for bullish V pattern
            bullish_pattern = self._check_bullish_v_pattern(df, i)
            if bullish_pattern:
                patterns.append(bullish_pattern)
            
            # Check for bearish V pattern
            bearish_pattern = self._check_bearish_v_pattern(df, i)
            if bearish_pattern:
                patterns.append(bearish_pattern)
        
        self.patterns = patterns
        return patterns
    
    def _check_bullish_v_pattern(self, df: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """Check for bullish V pattern starting at given index"""
        
        # L1: Single candle (must be red/bearish)
        l1_candle = df.iloc[start_idx]
        if l1_candle['Close'] >= l1_candle['Open']:  # Not bearish
            return None
        
        l1_low = l1_candle['Low']
        l1_high = l1_candle['High']
        l1_length = abs(l1_candle['Close'] - l1_candle['Open'])
        
        # L2: Check next 1-3 candles for bullish reversal
        for l2_length in range(1, 4):  # 1 to 3 candles
            if start_idx + l2_length >= len(df):
                continue
                
            l2_candles = df.iloc[start_idx + 1:start_idx + 1 + l2_length]
            
            # All L2 candles must be bullish (green)
            all_bullish = all(candle['Close'] > candle['Open'] for _, candle in l2_candles.iterrows())
            if not all_bullish:
                continue
            
            # Check breakout within first 2 candles of L2
            breakout_confirmed = False
            breakout_candle_idx = None
            
            for j in range(min(2, l2_length)):
                candle = l2_candles.iloc[j]
                if candle['Close'] > l1_high:  # Breakout above L1 high
                    breakout_confirmed = True
                    breakout_candle_idx = start_idx + 1 + j
                    break
            
            if not breakout_confirmed:
                continue
            
            # Check L2 length constraint (max 200% of L1)
            l2_total_length = sum(abs(candle['Close'] - candle['Open']) for _, candle in l2_candles.iterrows())
            if l2_total_length > l1_length * 2:  # 200% constraint
                continue
            
            # Check for confirmation candle (if available)
            confirmation_idx = start_idx + 1 + l2_length
            if confirmation_idx < len(df):
                confirmation_candle = df.iloc[confirmation_idx]
                # Confirmation candle should also be bullish
                if confirmation_candle['Close'] > confirmation_candle['Open']:
                    return {
                        'type': 'bullish',
                        'start_idx': start_idx,
                        'end_idx': confirmation_idx,
                        'l1_idx': start_idx,
                        'l2_start': start_idx + 1,
                        'l2_end': start_idx + l2_length,
                        'l2_length': l2_length,
                        'breakout_idx': breakout_candle_idx,
                        'confirmation_idx': confirmation_idx,
                        'l1_low': l1_low,
                        'l1_high': l1_high,
                        'breakout_price': df.iloc[breakout_candle_idx]['Close'],
                        'pattern_low': min(df.iloc[start_idx:confirmation_idx+1]['Low']),
                        'pattern_high': max(df.iloc[start_idx:confirmation_idx+1]['High']),
                        'timestamp': df.index[start_idx]
                    }
        
        return None
    
    def _check_bearish_v_pattern(self, df: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """Check for bearish V pattern starting at given index"""
        
        # L1: Single candle (must be green/bullish)
        l1_candle = df.iloc[start_idx]
        if l1_candle['Close'] <= l1_candle['Open']:  # Not bullish
            return None
        
        l1_low = l1_candle['Low']
        l1_high = l1_candle['High']
        l1_length = abs(l1_candle['Close'] - l1_candle['Open'])
        
        # L2: Check next 1-3 candles for bearish reversal
        for l2_length in range(1, 4):  # 1 to 3 candles
            if start_idx + l2_length >= len(df):
                continue
                
            l2_candles = df.iloc[start_idx + 1:start_idx + 1 + l2_length]
            
            # All L2 candles must be bearish (red)
            all_bearish = all(candle['Close'] < candle['Open'] for _, candle in l2_candles.iterrows())
            if not all_bearish:
                continue
            
            # Check breakout within first 2 candles of L2
            breakout_confirmed = False
            breakout_candle_idx = None
            
            for j in range(min(2, l2_length)):
                candle = l2_candles.iloc[j]
                if candle['Close'] < l1_low:  # Breakout below L1 low
                    breakout_confirmed = True
                    breakout_candle_idx = start_idx + 1 + j
                    break
            
            if not breakout_confirmed:
                continue
            
            # Check L2 length constraint (max 200% of L1)
            l2_total_length = sum(abs(candle['Close'] - candle['Open']) for _, candle in l2_candles.iterrows())
            if l2_total_length > l1_length * 2:  # 200% constraint
                continue
            
            # Check for confirmation candle (if available)
            confirmation_idx = start_idx + 1 + l2_length
            if confirmation_idx < len(df):
                confirmation_candle = df.iloc[confirmation_idx]
                # Confirmation candle should also be bearish
                if confirmation_candle['Close'] < confirmation_candle['Open']:
                    return {
                        'type': 'bearish',
                        'start_idx': start_idx,
                        'end_idx': confirmation_idx,
                        'l1_idx': start_idx,
                        'l2_start': start_idx + 1,
                        'l2_end': start_idx + l2_length,
                        'l2_length': l2_length,
                        'breakout_idx': breakout_candle_idx,
                        'confirmation_idx': confirmation_idx,
                        'l1_low': l1_low,
                        'l1_high': l1_high,
                        'breakout_price': df.iloc[breakout_candle_idx]['Close'],
                        'pattern_low': min(df.iloc[start_idx:confirmation_idx+1]['Low']),
                        'pattern_high': max(df.iloc[start_idx:confirmation_idx+1]['High']),
                        'timestamp': df.index[start_idx]
                    }
        
        return None
    
    def plot_patterns(self, df: pd.DataFrame, patterns: List[Dict], title: str = "V Pattern Detection"):
        """
        Plot candlestick chart with detected V patterns marked
        """
        plt.figure(figsize=(15, 8))
        
        # Plot candlesticks
        for i in range(len(df)):
            candle = df.iloc[i]
            color = 'green' if candle['Close'] > candle['Open'] else 'red'
            
            # Candle body
            plt.plot([i, i], [candle['Open'], candle['Close']], color=color, linewidth=3)
            # Candle wick
            plt.plot([i, i], [candle['Low'], candle['High']], color='black', linewidth=1)
        
        # Mark V patterns
        for pattern in patterns:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            if pattern['type'] == 'bullish':
                # Mark bullish V pattern
                plt.scatter(start_idx, pattern['l1_low'], color='lime', marker='^', s=100, label='Bullish V')
                plt.plot([start_idx, pattern['breakout_idx']], 
                        [pattern['l1_low'], pattern['breakout_price']], 
                        color='lime', linewidth=2, linestyle='--')
            else:
                # Mark bearish V pattern
                plt.scatter(start_idx, pattern['l1_high'], color='red', marker='v', s=100, label='Bearish V')
                plt.plot([start_idx, pattern['breakout_idx']], 
                        [pattern['l1_high'], pattern['breakout_price']], 
                        color='red', linewidth=2, linestyle='--')
        
        plt.title(title)
        plt.xlabel('Time Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class VPatternBacktester:
    """
    Backtesting system for V Pattern trading strategy
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.performance_stats = {}
    
    def backtest(self, df: pd.DataFrame, patterns: List[Dict], 
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04,
                 risk_per_trade: float = 0.02) -> Dict:
        """
        Backtest V pattern trading strategy
        
        Args:
            df: OHLCV DataFrame
            patterns: Detected V patterns
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            risk_per_trade: Risk per trade as percentage of capital
        
        Returns:
            Performance statistics dictionary
        """
        
        self.trades = []
        capital_history = [self.initial_capital]
        
        for pattern in patterns:
            entry_idx = pattern['confirmation_idx']
            if entry_idx >= len(df) - 10:  # Need some candles after entry
                continue
            
            entry_price = df.iloc[entry_idx]['Close']
            
            if pattern['type'] == 'bullish':
                # Long trade
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                direction = 'long'
            else:
                # Short trade
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                direction = 'short'
            
            # Calculate position size based on risk
            risk_amount = self.capital * risk_per_trade
            price_risk = abs(entry_price - stop_loss)
            position_size = risk_amount / price_risk if price_risk > 0 else 0
            
            if position_size <= 0:
                continue
            
            # Find exit point
            exit_idx, exit_price, exit_reason = self._find_exit_point(
                df, entry_idx, stop_loss, take_profit, direction
            )
            
            if exit_idx is None:
                continue
            
            # Calculate P&L
            if direction == 'long':
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            pnl_pct = pnl / (entry_price * position_size) * 100
            
            # Update capital
            self.capital += pnl
            capital_history.append(self.capital)
            
            # Record trade
            trade = {
                'pattern_type': pattern['type'],
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            self.trades.append(trade)
        
        # Calculate performance statistics
        self.performance_stats = self._calculate_performance_stats(capital_history)
        return self.performance_stats
    
    def _find_exit_point(self, df: pd.DataFrame, entry_idx: int, 
                        stop_loss: float, take_profit: float, direction: str) -> Tuple:
        """Find exit point for trade based on stop loss or take profit"""
        
        for i in range(entry_idx + 1, min(entry_idx + 50, len(df))):  # Max 50 candles
            candle = df.iloc[i]
            
            if direction == 'long':
                if candle['Low'] <= stop_loss:
                    return i, stop_loss, 'stop_loss'
                elif candle['High'] >= take_profit:
                    return i, take_profit, 'take_profit'
            else:  # short
                if candle['High'] >= stop_loss:
                    return i, stop_loss, 'stop_loss'
                elif candle['Low'] <= take_profit:
                    return i, take_profit, 'take_profit'
        
        # If no exit found, close at last available price
        return len(df) - 1, df.iloc[-1]['Close'], 'time_exit'
    
    def _calculate_performance_stats(self, capital_history: List[float]) -> Dict:
        """Calculate comprehensive performance statistics"""
        
        if not self.trades:
            return {}
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        # Drawdown calculation
        peak = self.initial_capital
        max_drawdown = 0
        for capital in capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Pattern-specific stats
        bullish_trades = [t for t in self.trades if t['pattern_type'] == 'bullish']
        bearish_trades = [t for t in self.trades if t['pattern_type'] == 'bearish']
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'final_capital': self.capital,
            'bullish_patterns': len(bullish_trades),
            'bearish_patterns': len(bearish_trades),
            'bullish_win_rate': len([t for t in bullish_trades if t['pnl'] > 0]) / len(bullish_trades) * 100 if bullish_trades else 0,
            'bearish_win_rate': len([t for t in bearish_trades if t['pnl'] > 0]) / len(bearish_trades) * 100 if bearish_trades else 0
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        
        if not self.performance_stats:
            return "No backtest results available. Run backtest first."
        
        stats = self.performance_stats
        
        report = f"""
=== V PATTERN TRADING STRATEGY - PERFORMANCE REPORT ===

OVERALL PERFORMANCE:
├─ Total Trades: {stats['total_trades']}
├─ Win Rate: {stats['win_rate']:.2f}% ({stats['winning_trades']}/{stats['total_trades']})
├─ Total Return: {stats['total_return_pct']:.2f}%
├─ Final Capital: ${stats['final_capital']:,.2f}
├─ Total P&L: ${stats['total_pnl']:,.2f}
└─ Max Drawdown: {stats['max_drawdown_pct']:.2f}%

TRADE ANALYSIS:
├─ Average Win: ${stats['avg_win']:,.2f}
├─ Average Loss: ${stats['avg_loss']:,.2f}
├─ Profit Factor: {stats['profit_factor']:.2f}
├─ Winning Trades: {stats['winning_trades']}
└─ Losing Trades: {stats['losing_trades']}

PATTERN BREAKDOWN:
├─ Bullish V Patterns: {stats['bullish_patterns']} (Win Rate: {stats['bullish_win_rate']:.2f}%)
└─ Bearish V Patterns: {stats['bearish_patterns']} (Win Rate: {stats['bearish_win_rate']:.2f}%)

RISK METRICS:
├─ Profit Factor: {stats['profit_factor']:.2f}
├─ Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%
└─ Risk-Adjusted Return: {stats['total_return_pct'] / max(stats['max_drawdown_pct'], 1):.2f}

========================================================
        """
        
        return report

def get_bitcoin_data(period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """
    Fetch Bitcoin data from Yahoo Finance
    
    Args:
        period: Data period (60d for 60 days - max for 15m interval)
        interval: Data interval (15m for 15 minutes)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period=period, interval=interval)
        
        # Clean and prepare data
        data = data.dropna()
        
        # Yahoo Finance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # We only need OHLCV, so select and rename appropriately
        if len(data.columns) >= 5:
            data = data.iloc[:, :5]  # Take first 5 columns
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"✅ Data shape: {data.shape}")
        print(f"✅ Columns: {list(data.columns)}")
        
        return data
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("💡 Trying alternative approach...")
        
        # Alternative: Try 1-hour data for longer period
        try:
            data = ticker.history(period="6mo", interval="1h")
            data = data.dropna()
            if len(data.columns) >= 5:
                data = data.iloc[:, :5]
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            print(f"✅ Alternative data fetched: {data.shape} (1-hour intervals)")
            return data
        except:
            return pd.DataFrame()

# Example usage and main execution
def main():
    """Main execution function demonstrating the V Pattern Detection system"""
    
    print("🚀 V Pattern Detection & Backtesting System")
    print("=" * 50)
    
    # 1. Fetch Bitcoin data (15min for 60 days, or 1h for 6 months)
    print("📊 Fetching Bitcoin data...")
    df = get_bitcoin_data(period="60d", interval="15m")
    
    if df.empty:
        print("❌ Failed to fetch data. Please check your internet connection.")
        return
    
    print(f"✅ Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # 2. Initialize V Pattern Detector
    detector = VPatternDetector()
    
    # 3. Detect V patterns
    print("\n🔍 Detecting V patterns...")
    patterns = detector.detect_v_patterns(df)
    
    bullish_patterns = [p for p in patterns if p['type'] == 'bullish']
    bearish_patterns = [p for p in patterns if p['type'] == 'bearish']
    
    print(f"✅ Found {len(patterns)} V patterns:")
    print(f"   🟢 Bullish: {len(bullish_patterns)}")
    print(f"   🔴 Bearish: {len(bearish_patterns)}")
    
    # 4. Run backtest
    print("\n📈 Running backtest...")
    backtester = VPatternBacktester(initial_capital=10000)
    
    performance = backtester.backtest(
        df=df,
        patterns=patterns,
        stop_loss_pct=0.02,      # 2% stop loss
        take_profit_pct=0.04,    # 4% take profit
        risk_per_trade=0.02      # Risk 2% of capital per trade
    )
    
    # 5. Generate and display performance report
    print("\n📊 PERFORMANCE REPORT:")
    print(backtester.generate_report())
    
    # 6. Plot patterns (optional - comment out if running in non-GUI environment)
    try:
        # Plot last 500 candles with patterns for visualization
        recent_df = df.tail(500)
        recent_patterns = [p for p in patterns if p['start_idx'] >= len(df) - 500]
        
        # Adjust indices for recent_df
        for pattern in recent_patterns:
            pattern['start_idx'] -= (len(df) - 500)
            pattern['end_idx'] -= (len(df) - 500)
            pattern['breakout_idx'] -= (len(df) - 500)
            pattern['confirmation_idx'] -= (len(df) - 500)
        
        detector.plot_patterns(recent_df, recent_patterns, 
                             title="Bitcoin 15min - V Pattern Detection (Last 500 candles)")
    except Exception as e:
        print(f"⚠️  Plotting skipped: {e}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()