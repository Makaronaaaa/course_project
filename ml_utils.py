import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

class MarketPredictor:
    def __init__(self, window_size: int = 100, use_ml: bool = True):
        self.window_size = window_size
        self.use_ml = use_ml
        
        self.price_history = deque(maxlen=window_size)
        self.returns_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        self.order_imbalance_history = deque(maxlen=window_size)
        
        self.features = {
            'ma_short': [],
            'ma_medium': [],
            'ma_long': [],
            'vol_ewma': [],
            'rsi': [],
            'momentum': [],
            'volume_trend': []
        }
        
        self.is_trained = False
        self.prediction_error_history = deque(maxlen=100)
        self.last_prediction = 0
    
    def update(self, market_state: Dict) -> None:
        price = market_state['true_price']
        volume = market_state['trading_volume']
        volatility = market_state['volatility']
        order_imbalance = market_state.get('order_imbalance', 0)
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)
        self.order_imbalance_history.append(order_imbalance)
        
        if len(self.price_history) >= 2:
            ret = (price / list(self.price_history)[-2]) - 1
            self.returns_history.append(ret)
        
        if len(self.price_history) >= 50:
            self._update_features()
        
        if self.last_prediction > 0:
            error = abs(self.last_prediction - price) / price
            self.prediction_error_history.append(error)
    
    def _update_features(self) -> None:
        prices = list(self.price_history)
        returns = list(self.returns_history)
        volumes = list(self.volume_history)
        
        self.features['ma_short'] = np.mean(prices[-10:])
        self.features['ma_medium'] = np.mean(prices[-30:])
        self.features['ma_long'] = np.mean(prices[-50:])
        
        alpha = 0.1
        if not self.features['vol_ewma']:
            self.features['vol_ewma'] = np.std(returns[-20:])
        else:
            latest_vol = np.std(returns[-20:])
            self.features['vol_ewma'] = alpha * latest_vol + (1 - alpha) * self.features['vol_ewma']
        
        if len(returns) >= 14:
            gains = [r if r > 0 else 0 for r in returns[-14:]]
            losses = [abs(r) if r < 0 else 0 for r in returns[-14:]]
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                self.features['rsi'] = 100
            else:
                rs = avg_gain / avg_loss
                self.features['rsi'] = 100 - (100 / (1 + rs))
        else:
            self.features['rsi'] = 50
        
        if len(prices) >= 5:
            self.features['momentum'] = prices[-1] / prices[-5] - 1
        else:
            self.features['momentum'] = 0
        
        if len(volumes) >= 10:
            vol_ma_short = np.mean(volumes[-5:])
            vol_ma_long = np.mean(volumes[-10:])
            self.features['volume_trend'] = vol_ma_short / vol_ma_long - 1 if vol_ma_long > 0 else 0
        else:
            self.features['volume_trend'] = 0
    
    def predict_price(self, horizon: int = 1) -> float:
        if not self.use_ml or len(self.price_history) < 50:
            return list(self.price_history)[-1] if self.price_history else 0
        
        current_price = list(self.price_history)[-1]
        
        ma_signal = (self.features['ma_short'] / self.features['ma_long'] - 1) * 0.3
        momentum_signal = self.features['momentum'] * 0.4
        rsi_signal = (self.features['rsi'] - 50) / 50 * 0.2
        volume_signal = self.features['volume_trend'] * 0.1
        
        mean_reversion = (self.features['ma_medium'] / current_price - 1) * 0.2
        
        expected_return = ma_signal + momentum_signal + rsi_signal + volume_signal + mean_reversion
        
        vol_scale = self.features['vol_ewma'] * math.sqrt(horizon)
        expected_return *= vol_scale
        
        expected_return = np.clip(expected_return, -0.05, 0.05)
        
        prediction = current_price * (1 + expected_return)
        self.last_prediction = prediction
        
        return prediction
    
    def predict_volatility(self, horizon: int = 1) -> float:
        if not self.use_ml or len(self.volatility_history) < 20:
            return list(self.volatility_history)[-1] if self.volatility_history else 0.01
        
        vol_history = list(self.volatility_history)[-20:]
        returns = list(self.returns_history)[-20:]
        
        base_vol = vol_history[-1]
        
        recent_vol_change = vol_history[-1] / vol_history[-5] if len(vol_history) >= 5 else 1
        
        return_deviation = 0
        if returns:
            return_deviation = abs(returns[-1]) / np.std(returns) if np.std(returns) > 0 else 0
        
        vol_prediction = base_vol * (
            0.8 +
            0.1 * recent_vol_change +
            0.1 * min(3, return_deviation)
        )
        
        vol_prediction *= math.sqrt(horizon)
        
        return vol_prediction
    
    def get_prediction_accuracy(self) -> float:
        if not self.prediction_error_history:
            return 0
        
        avg_error = np.mean(list(self.prediction_error_history))
        
        return max(0, 1 - avg_error * 10)