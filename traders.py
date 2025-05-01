import numpy as np
from typing import Dict, Tuple, List
import math

class Trader:
    def __init__(self, name: str, realistic_behavior: bool = False):
        self.name = name
        self.position = 0
        self.cash = 0
        self.realistic_behavior = realistic_behavior
        self.trades_history = []
        self.realized_pnl = 0
        self.risk_aversion = 0.1

    def place_order(self, bid_price: float, ask_price: float, market_state: Dict) -> Tuple[int, float]:
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_pnl(self, current_price: float) -> float:
        mark_to_market = self.position * current_price
        return self.cash + mark_to_market

    def _apply_realistic_behavior(self, base_size: int, price: float, market_state: Dict) -> int:
        if not self.realistic_behavior:
            return base_size
        
        if base_size == 0:
            return 0
            
        current_price = market_state.get("true_price", price)
        
        current_pnl = self.calculate_pnl(current_price)
        pnl_factor = 1.0
        if current_pnl < 0:
            pnl_factor = max(0.5, 1.0 + current_pnl / (abs(self.position) * current_price) if self.position != 0 else 1.0)
        
        momentum_factor = 1.0
        if "market_regime" in market_state and market_state["market_regime"] == "trending":
            trend_direction = 1 if market_state.get("order_imbalance", 0) > 0 else -1
            order_direction = 1 if base_size > 0 else -1
            
            if trend_direction == order_direction:
                momentum_factor = 1.2
            else:
                momentum_factor = 0.8
        
        position_factor = 1.0
        max_position = 100
        if abs(self.position) > max_position * 0.7:
            position_factor = max(0.3, 1.0 - abs(self.position) / max_position)
        
        herding_factor = 1.0
        if "market_regime" in market_state:
            regime = market_state["market_regime"]
            if regime == "crisis":
                herding_factor = 0.7 if base_size > 0 else 1.3
            elif regime == "volatile":
                herding_factor = np.random.normal(1.0, 0.2)
        
        adjusted_size = int(base_size * pnl_factor * momentum_factor * position_factor * herding_factor)
        
        if base_size != 0 and adjusted_size == 0:
            adjusted_size = 1 if base_size > 0 else -1
            
        return adjusted_size


class InformedTrader(Trader):
    def __init__(self, name: str, information_advantage: float = 0.2, 
                 max_position: int = 100, realistic_behavior: bool = False,
                 confidence_variation: bool = True):
        super().__init__(name, realistic_behavior)
        self.information_advantage = information_advantage
        self.max_position = max_position
        self.confidence_variation = confidence_variation
        self.confidence = 1.0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.last_prediction = 0
        self.last_price = 0
        
        self.risk_tolerance = np.random.normal(1.0, 0.2)
        self.overconfidence = np.random.normal(1.2, 0.3)
        self.recency_bias = np.random.normal(0.7, 0.2)

    def place_order(self, bid_price: float, ask_price: float, market_state: Dict) -> Tuple[int, float]:
        current_step = market_state["time_step"]
        true_price = market_state["true_price"]
        
        if self.last_price > 0 and self.confidence_variation:
            self._update_confidence(true_price)
        
        self.last_price = true_price
        
        look_ahead = min(
            int(10 * self.information_advantage),
            len(market_state["future_prices"]) - current_step - 1 if "future_prices" in market_state else 10
        )
        
        if look_ahead <= 0:
            return 0, 0
        
        if "future_prices" in market_state:
            future_price = market_state["future_prices"][current_step + look_ahead]
            noise_factor = (1.0 - self.confidence) * 0.05
            future_price *= np.random.normal(1.0, noise_factor)
        else:
            vol_estimate = market_state.get("volatility", 0.01)
            future_price = true_price * (1 + np.random.normal(0, vol_estimate) * look_ahead)
        
        self.last_prediction = future_price
        self.total_predictions += 1
        
        if future_price > ask_price * (1 + 0.001) and self.position < self.max_position:
            expected_return = (future_price / ask_price) - 1
            base_size = self._calculate_optimal_size(expected_return, "buy")
            base_size = min(base_size, self.max_position - self.position)
            
            size = self._apply_realistic_behavior(base_size, ask_price, market_state)
            
            if size > 0:
                execution_price = ask_price
                self.position += size
                self.cash -= size * execution_price
                self.trades_history.append(("buy", size, execution_price))
                return size, execution_price

        elif future_price < bid_price * (1 - 0.001) and self.position > -self.max_position:
            expected_return = 1 - (future_price / bid_price)
            base_size = self._calculate_optimal_size(expected_return, "sell")
            base_size = min(base_size, self.max_position + self.position)
            
            size = self._apply_realistic_behavior(base_size, bid_price, market_state)
            
            if size > 0:
                execution_price = bid_price
                self.position -= size
                self.cash += size * execution_price
                self.trades_history.append(("sell", size, execution_price))
                return -size, execution_price

        return 0, 0

    def _calculate_optimal_size(self, expected_return: float, direction: str) -> int:
        base_size = int(np.ceil(expected_return * 20 * self.max_position))
        
        confidence_factor = self.confidence * self.overconfidence
        
        risk_factor = self.risk_tolerance
        
        size = int(max(1, base_size * confidence_factor * risk_factor))
        
        return min(max(1, size), int(self.max_position * 0.5))

    def _update_confidence(self, current_price: float) -> None:
        if self.last_prediction > 0:
            prediction_error = abs(self.last_prediction - current_price) / current_price
            
            if prediction_error < 0.01:
                self.correct_predictions += 1
                self.confidence = min(1.0, self.confidence + 0.05 * self.recency_bias)
            else:
                self.confidence = max(0.3, self.confidence - 0.1 * prediction_error * self.recency_bias)


class LiquidityTrader(Trader):
    def __init__(self, name: str, trade_probability: float = 0.3, 
                 max_size: int = 5, realistic_behavior: bool = False,
                 time_sensitive: bool = True):
        super().__init__(name, realistic_behavior)
        self.trade_probability = trade_probability
        self.max_size = max_size
        self.time_sensitive = time_sensitive
        
        self.order_imbalance_sensitivity = np.random.normal(0.5, 0.2)
        self.volatility_aversion = np.random.normal(1.0, 0.3)
        self.time_of_day_pattern = np.random.choice(["morning", "midday", "afternoon", "uniform"])

    def place_order(self, bid_price: float, ask_price: float, market_state: Dict) -> Tuple[int, float]:
        effective_probability = self.trade_probability
        
        if self.time_sensitive and "time_of_day" in market_state:
            time_factor = self._time_of_day_factor(market_state["time_of_day"])
            effective_probability *= time_factor
        
        if "market_regime" in market_state:
            regime = market_state["market_regime"]
            if regime == "crisis":
                effective_probability *= 1.5
            elif regime == "volatile":
                effective_probability *= 1.2
        
        if np.random.random() > effective_probability:
            return 0, 0
        
        base_buy_probability = 0.5
        
        if "order_imbalance" in market_state:
            imbalance = market_state["order_imbalance"]
            adjustment = -imbalance * 0.2 * self.order_imbalance_sensitivity
            base_buy_probability += adjustment
        
        is_buy = np.random.random() < base_buy_probability
        
        size_factor = 1.0
        if "volatility" in market_state:
            vol = market_state["volatility"]
            base_vol = 0.01
            size_factor = (base_vol / vol) ** self.volatility_aversion if vol > 0 else 1.0
            size_factor = np.clip(size_factor, 0.5, 2.0)
        
        base_size = np.random.randint(1, max(2, int(self.max_size * size_factor) + 1))
        
        size = self._apply_realistic_behavior(base_size, bid_price if not is_buy else ask_price, market_state)
        
        if is_buy:
            execution_price = ask_price
            self.position += size
            self.cash -= size * execution_price
            return size, execution_price
        else:
            execution_price = bid_price
            self.position -= size
            self.cash += size * execution_price
            return -size, execution_price

    def _time_of_day_factor(self, time_of_day: float) -> float:
        if self.time_of_day_pattern == "morning":
            return 1.5 * (1 - time_of_day) + 0.5
        elif self.time_of_day_pattern == "midday":
            return 1.5 - 2.0 * abs(time_of_day - 0.5)
        elif self.time_of_day_pattern == "afternoon":
            return 1.5 * time_of_day + 0.5
        else:
            return 1.0