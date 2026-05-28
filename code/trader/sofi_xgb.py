from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "code" / "trader" / "sofi_xgb_model.pkl"
METADATA_PATH = PROJECT_ROOT / "code" / "trader" / "sofi_xgb_model_meta.json"
FEATURE_COLUMNS = [
    "rsi_14", "rsi_7", "macd_line", "macd_signal", "macd_hist",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema9", "price_vs_ema21",
    "price_vs_vwap", "atr_pct", "returns_5", "returns_10", "returns_20",
    "vol_ratio", "price_position", "body_pct", "upper_wick", "lower_wick",
    "time_of_day",
]


@dataclass
class SofiXGBPrediction:
    trend: str
    confidence: float
    probabilities: dict[str, float]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"])
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    open_price = data["open"]

    data["rsi_14"] = _rsi(close, 14)
    data["rsi_7"] = _rsi(close, 7)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd_line"] = ema12 - ema26
    data["macd_signal"] = data["macd_line"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd_line"] - data["macd_signal"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    data["ema_cross_9_21"] = (ema9 > ema21).astype(int) * 2 - 1
    data["ema_cross_21_50"] = (ema21 > ema50).astype(int) * 2 - 1
    data["price_vs_ema9"] = (close - ema9) / close * 100
    data["price_vs_ema21"] = (close - ema21) / close * 100

    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
    data["price_vs_vwap"] = (close - vwap) / close * 100

    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = tr.rolling(14).mean()
    data["atr_pct"] = atr / close * 100

    data["returns_5"] = close.pct_change(5) * 100
    data["returns_10"] = close.pct_change(10) * 100
    data["returns_20"] = close.pct_change(20) * 100

    vol_avg = volume.rolling(20).mean()
    data["vol_ratio"] = volume / (vol_avg + 1e-10)

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    data["price_position"] = (close - low20) / (high20 - low20 + 1e-10)

    data["body_pct"] = (close - open_price) / close * 100
    data["upper_wick"] = (high - np.maximum(close, open_price)) / close * 100
    data["lower_wick"] = (np.minimum(close, open_price) - low) / close * 100

    if hasattr(data.index, "hour"):
        data["time_of_day"] = data.index.hour + data.index.minute / 60
    else:
        data["time_of_day"] = 12.0

    return data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def make_labels(df: pd.DataFrame, horizon_bars: int = 18, threshold_pct: float = 0.35) -> pd.Series:
    future_return = (df["close"].shift(-horizon_bars) / df["close"] - 1) * 100
    labels = pd.Series(1, index=df.index)  # chop
    labels[future_return > threshold_pct] = 2  # bull
    labels[future_return < -threshold_pct] = 0  # bear
    return labels


def train_model(df: pd.DataFrame, horizon_bars: int = 18, threshold_pct: float = 0.35) -> dict:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    features = extract_features(df)
    labels = make_labels(df, horizon_bars=horizon_bars, threshold_pct=threshold_pct)
    valid = features.notna().all(axis=1) & labels.notna()
    X = features[valid]
    y = labels[valid].astype(int)
    if len(X) < 300:
        raise RuntimeError(f"not enough SOFI training rows: {len(X)}")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=76,
    )
    model.fit(X_train, y_train)
    accuracy = float((model.predict(X_test) == y_test).mean()) if len(X_test) else 0.0
    class_counts = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    metadata = {
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": accuracy,
        "class_counts": class_counts,
        "horizon_bars": horizon_bars,
        "threshold_pct": threshold_pct,
        "features": FEATURE_COLUMNS,
        "start": df.index.min().isoformat() if len(df) else None,
        "end": df.index.max().isoformat() if len(df) else None,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_latest(df: pd.DataFrame) -> SofiXGBPrediction | None:
    model = load_model()
    if model is None or df.empty:
        return None
    features = extract_features(df).dropna()
    if features.empty:
        return None
    probs = model.predict_proba(features.iloc[[-1]])[0]
    labels = ["bear", "chop", "bull"]
    idx = int(np.argmax(probs))
    return SofiXGBPrediction(
        trend=labels[idx],
        confidence=float(probs[idx]),
        probabilities={label: float(prob) for label, prob in zip(labels, probs)},
    )
