# trader/indicators.py - Full fixed version with get_trend_signal included
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TrainedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

_lstm_model = TrainedLSTM()
_lstm_model.eval()
_optimizer = None
_criterion = nn.MSELoss()
_last_training_date = None

def _calculate_vwap(df: pd.DataFrame) -> float:
    if df is None or df.empty: return 0.0
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vol_sum = df['volume'].sum()
    if vol_sum == 0: return typical_price.iloc[-1]
    return (typical_price * df['volume']).cumsum().iloc[-1] / vol_sum

def _ema(series: pd.Series, span: int) -> float:
    if series is None or len(series) < span: return 0.0
    return series.ewm(span=span, adjust=False).mean().iloc[-1]

def _rsi(series: pd.Series, period: int = 14) -> float:
    if series is None or len(series) <= period: return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    avg_loss = loss.iloc[-1]
    if avg_loss == 0: return 100.0
    rs = gain.iloc[-1] / avg_loss
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series) -> tuple:
    if series is None or len(series) < 26: return 0.0, 0.0
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) <= period: return 0.0
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, high_close, low_close)
    return tr.rolling(window=period).mean().iloc[-1]

def _lstm_predict(series: pd.Series, seq_len: int = 60, steps_ahead: int = 5) -> str:
    global _lstm_model
    if len(series) < seq_len + 1:
        return "chop"
    try:
        data = series.values.astype(np.float32)
        window = data[-seq_len*2:]
        min_val, max_val = np.min(window), np.max(window)
        if max_val - min_val == 0:
            return "chop"
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        
        input_seq = normalized[-seq_len:].reshape(1, seq_len, 1)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        
        preds = []
        current_tensor = input_tensor.clone()  # Important: clone to avoid inplace issues
        with torch.no_grad():
            for _ in range(steps_ahead):
                pred_tensor = _lstm_model(current_tensor)
                pred_norm = pred_tensor.item()
                preds.append(pred_norm)
                # Append prediction correctly
                pred_append = pred_tensor.unsqueeze(1)  # (1,1,1)
                current_tensor = torch.cat((current_tensor[:, 1:, :], pred_append), dim=1)
        
        avg_pred = np.mean(preds)
        current_norm = normalized[-1]
        if avg_pred > current_norm + 0.005:
            return "bull"
        elif avg_pred < current_norm - 0.005:
            return "bear"
        return "chop"
    except Exception as e:
        print(f"[LSTM Predict Error] {e}")
        return "chop"

def train_lstm_daily(bars_1m: pd.DataFrame, seq_len: int = 60, epochs: int = 8):
    global _lstm_model, _optimizer, _last_training_date
    
    today = datetime.now().date()
    if _last_training_date == today:
        return
    
    if len(bars_1m) < seq_len * 30:
        print("[LSTM] Not enough bars for training today")
        return
    
    close = bars_1m['close'].values.astype(np.float32)
    train_window = close[-seq_len*150:]
    min_v, max_v = train_window.min(), train_window.max()
    if max_v - min_v == 0:
        return
    normalized = (train_window - min_v) / (max_v - min_v + 1e-8)
    
    X, y = [], []
    for i in range(len(normalized) - seq_len - 5):
        X.append(normalized[i:i+seq_len])
        y.append(normalized[i+seq_len:i+seq_len+5].mean())
    
    X = torch.tensor(np.array(X)).unsqueeze(-1)
    y = torch.tensor(np.array(y)).unsqueeze(-1)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    _optimizer = optim.Adam(_lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    _lstm_model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            _optimizer.zero_grad()
            pred = _lstm_model(batch_x)
            loss = _criterion(pred, batch_y)
            loss.backward()
            _optimizer.step()
            epoch_loss += loss.item()
        print(f"[LSTM] Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader):.6f}")
    
    _lstm_model.eval()
    _last_training_date = today
    print(f"[LSTM] Daily training complete on {len(bars_1m)} bars")

def get_trend_signal(bars_1m: pd.DataFrame, bars_5m: pd.DataFrame = None, bars_15m: pd.DataFrame = None) -> str:
    if bars_1m is None or bars_1m.empty:
        return "chop"
  
    valid_tfs = [tf for tf in [bars_1m, bars_5m, bars_15m] if tf is not None and not tf.empty]
    if not valid_tfs:
        return "chop"
  
    price = bars_1m['close'].iloc[-1]
    num_tfs = len(valid_tfs)
  
    vwap_votes = sum(price > _calculate_vwap(tf) for tf in valid_tfs)
  
    ema_tfs = [tf for tf in valid_tfs if len(tf) >= 21]
    ema_votes = sum(_ema(tf['close'], 9) > _ema(tf['close'], 21) for tf in ema_tfs)
  
    macd_tfs = [tf for tf in valid_tfs if len(tf) >= 26]
    macd_votes = sum(_macd(tf['close'])[0] > _macd(tf['close'])[1] for tf in macd_tfs)
  
    rsi_subset = [tf for tf in [bars_1m, bars_5m] if tf is not None and len(tf) > 14]
    rsi_votes = sum(1 if _rsi(tf['close']) > 50 else -1 if _rsi(tf['close']) < 50 else 0 for tf in rsi_subset)
  
    atr_1m = _atr(bars_1m) if len(bars_1m) >= 14 else 0.0
  
    lstm_pred = _lstm_predict(bars_1m['close'])
    lstm_vote = 1 if lstm_pred == "bull" else -1 if lstm_pred == "bear" else 0
  
    bull_score = (vwap_votes / num_tfs if num_tfs else 0) + \
                 (ema_votes / len(ema_tfs) if ema_tfs else 0) + \
                 1.5 * (macd_votes / len(macd_tfs) if macd_tfs else 0) + \
                 (rsi_votes > 0) + \
                 1.2 * lstm_vote if lstm_vote != 0 else 0
                
    bear_score = ((num_tfs - vwap_votes) / num_tfs if num_tfs else 0) + \
                 ((len(ema_tfs) - ema_votes) / len(ema_tfs) if ema_tfs else 0) + \
                 1.5 * ((len(macd_tfs) - macd_votes) / len(macd_tfs) if macd_tfs else 0) + \
                 (rsi_votes < 0) + \
                 1.2 * (lstm_vote < 0)
  
    if bull_score >= 3.5 or (bull_score >= 3 and atr_1m < price * 0.003):
        return "bull"
    elif bear_score >= 3.5 or (bear_score >= 3 and atr_1m < price * 0.003):
        return "bear"
    else:
        return "chop"