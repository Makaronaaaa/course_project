import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque

class MarketMaker:
    def __init__(self, name: str, initial_capital: float = 100000,
                 max_position: int = 100, strategy_type: str = "adaptive",
                 base_spread: float = 0.002, spread_multiplier: float = 5.0,
                 position_limit_pct: float = 0.8, risk_aversion: float = 0.1,
                 information_risk_management: bool = False,
                 inventory_half_life: float = 60.0,
                 use_ml_prediction: bool = False):

        self.name = name
        self.capital = initial_capital
        self.max_position = max_position
        self.strategy_type = strategy_type
        self.base_spread = base_spread
        self.spread_multiplier = spread_multiplier
        self.position_limit_pct = position_limit_pct
        self.risk_aversion = risk_aversion
        self.information_risk_management = information_risk_management
        self.inventory_half_life = inventory_half_life
        self.use_ml_prediction = use_ml_prediction

        self.position = 0
        self.cash = initial_capital
        self.current_bid = 0
        self.current_ask = 0
        self.trades_executed = 0
        self.inventory_cost = 0
        self.trade_sizes = []
        self.average_position_cost = 0.0
        self.last_trade_time = 0
        self.market_impact = 0.0
        
        self.var_95 = 0.0
        self.max_drawdown = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        self.toxic_flow_metric = 0.0
        self.recent_trades = deque(maxlen=50)
        self.informed_trader_probability = dict()
        
        if use_ml_prediction:
            self.price_history = deque(maxlen=100)
            self.returns_history = deque(maxlen=100)
            self.volume_history = deque(maxlen=100)
            self.volatility_history = deque(maxlen=100)
            self.price_prediction = 0.0
            self.volatility_prediction = 0.0

        self.history = {
            "time": [],
            "position": [],
            "bid_price": [],
            "ask_price": [],
            "mid_price": [],
            "spread": [],
            "true_price": [],
            "capital": [],
            "pnl": [],
            "var_95": [],
            "toxic_flow": [],
            "predicted_price": [],
            "price_error": [],
            "market_regime": []
        }

    def _calculate_passive_spread(self, price: float, volatility: float) -> float:
        return self.base_spread * price

    def _calculate_aggressive_spread(self, price: float, volatility: float) -> float:
        return self.base_spread * self.spread_multiplier * price / 2

    def _calculate_adaptive_spread(self, price: float, volatility: float, volume: int, 
                                 market_state: Dict) -> float:
        vol_component = 1 + (volatility / 0.01) * 2
        
        volume_factor = max(0.5, 1 - (volume / 100) * 0.5)
        
        position_pct = abs(self.position) / self.max_position
        position_factor = 1 + position_pct * 2
        
        time_factor = 1.0
        if "time_of_day" in market_state:
            t = market_state["time_of_day"]
            time_factor = 1.0 + 0.5 * (np.sin(np.pi * (1 - t)) ** 2)
        
        regime_factor = 1.0
        if "market_regime" in market_state:
            regime = market_state.get("market_regime", "normal")
            if regime == "crisis":
                regime_factor = 3.0
            elif regime == "volatile":
                regime_factor = 2.0
            elif regime == "trending":
                regime_factor = 1.2
        
        info_risk_factor = 1.0
        if self.information_risk_management and self.toxic_flow_metric > 0:
            info_risk_factor = 1.0 + self.toxic_flow_metric
        
        spread = (self.base_spread * price * vol_component * volume_factor *
                 position_factor * time_factor * regime_factor * info_risk_factor)
        
        return spread

    def _calculate_ml_enhanced_spread(self, price: float, volatility: float, 
                                    market_state: Dict) -> float:
        if not self.use_ml_prediction:
            return self._calculate_adaptive_spread(price, volatility, 
                                                 market_state.get("trading_volume", 50),
                                                 market_state)
        
        base_spread = self._calculate_adaptive_spread(
            price, volatility, market_state.get("trading_volume", 50), market_state
        )
        
        expected_return = (self.price_prediction / price) - 1 if self.price_prediction > 0 else 0
        
        prediction_adjustment = abs(expected_return) * 10
        
        vol_adjustment = (self.volatility_prediction / volatility) if volatility > 0 else 1
        
        return base_spread * vol_adjustment * (1 + prediction_adjustment)

    def _adjust_quotes_for_inventory(self, mid_price: float, half_spread: float) -> Tuple[float, float]:
        position_pct = self.position / self.max_position
        max_adjustment = half_spread * 0.5
        
        adjustment = max_adjustment * (2 / (1 + np.exp(-5 * position_pct)) - 1)
        
        adjusted_mid = mid_price - adjustment
        
        bid_price = adjusted_mid - half_spread
        ask_price = adjusted_mid + half_spread
        
        return bid_price, ask_price

    def calculate_quotes(self, market_state: Dict) -> Tuple[float, float]:
        true_price = market_state["true_price"]
        volatility = market_state["volatility"]
        volume = market_state["trading_volume"]
        
        if self.use_ml_prediction:
            self._update_ml_predictions(market_state)
        
        if self.information_risk_management:
            self._update_information_risk(market_state)
        
        if self.strategy_type == "passive":
            spread = self._calculate_passive_spread(true_price, volatility)
        elif self.strategy_type == "aggressive":
            spread = self._calculate_aggressive_spread(true_price, volatility)
        elif self.strategy_type == "ml_enhanced" and self.use_ml_prediction:
            spread = self._calculate_ml_enhanced_spread(true_price, volatility, market_state)
        else:
            spread = self._calculate_adaptive_spread(true_price, volatility, volume, market_state)

        mid_price = true_price
        half_spread = spread / 2
        
        bid_price, ask_price = self._adjust_quotes_for_inventory(mid_price, half_spread)
        
        self.current_bid = bid_price
        self.current_ask = ask_price
        
        self._update_risk_metrics(true_price, volatility)
        
        return bid_price, ask_price

    def _update_information_risk(self, market_state: Dict) -> None:
        if not self.recent_trades:
            self.toxic_flow_metric = 0.0
            return
            
        buys = sum(1 for trade in self.recent_trades if trade > 0)
        sells = sum(1 for trade in self.recent_trades if trade < 0)
        
        if buys + sells == 0:
            self.toxic_flow_metric = 0.0
        else:
            imbalance = abs(buys - sells) / (buys + sells)
            time_decay = np.exp(-0.1 * (market_state["time_step"] - self.last_trade_time))
            
            self.toxic_flow_metric = 0.7 * self.toxic_flow_metric + 0.3 * imbalance * time_decay

    def _update_ml_predictions(self, market_state: Dict) -> None:
        true_price = market_state["true_price"]
        
        self.price_history.append(true_price)
        
        if len(self.price_history) > 1:
            latest_return = (true_price / self.price_history[-2]) - 1
            self.returns_history.append(latest_return)
        
        self.volume_history.append(market_state["trading_volume"])
        self.volatility_history.append(market_state["volatility"])
        
        if len(self.price_history) < 10:
            self.price_prediction = true_price
            self.volatility_prediction = market_state["volatility"]
            return
            
        short_ma = np.mean(list(self.price_history)[-5:])
        long_ma = np.mean(list(self.price_history)[-20:])
        momentum = short_ma / long_ma - 1
        
        self.price_prediction = true_price * (1 + 0.5 * momentum)
        
        alpha = 0.1
        if self.volatility_prediction == 0:
            self.volatility_prediction = market_state["volatility"]
        else:
            self.volatility_prediction = (
                alpha * market_state["volatility"] + (1 - alpha) * self.volatility_prediction
            )

    def _update_risk_metrics(self, price: float, volatility: float) -> None:
        position_value = self.position * price
        
        if self.position != 0 and self.average_position_cost > 0:
            self.unrealized_pnl = self.position * (price - self.average_position_cost)
        
        daily_var = abs(position_value) * volatility * 1.96 * np.sqrt(1/252)
        self.var_95 = daily_var
        
        total_pnl = self.realized_pnl + self.unrealized_pnl
        if total_pnl < self.max_drawdown:
            self.max_drawdown = total_pnl

    def execute_trade(self, order_size: int, price: float) -> None:
        if order_size == 0:
            return

        if self.position == 0:
            self.average_position_cost = price
        else:
            if (order_size > 0 and self.position > 0) or (order_size < 0 and self.position < 0):
                total_position = self.position + order_size
                self.average_position_cost = (
                    (self.average_position_cost * self.position + price * order_size) / total_position
                )

        old_position = self.position
        self.position += order_size
        self.cash -= order_size * price
        
        if (old_position > 0 and order_size < 0) or (old_position < 0 and order_size > 0):
            size_closed = min(abs(old_position), abs(order_size))
            trade_pnl = size_closed * (price - self.average_position_cost) * (1 if old_position > 0 else -1)
            self.realized_pnl += trade_pnl

        self.trades_executed += 1
        self.trade_sizes.append(abs(order_size))
        
        self.recent_trades.append(order_size)
        
        if abs(self.position) > self.max_position * self.position_limit_pct:
            excess = abs(self.position) - (self.max_position * self.position_limit_pct)
            self.inventory_cost = (excess / self.max_position) ** 2
        else:
            self.inventory_cost = 0

    def calculate_pnl(self, current_price: float) -> float:
        position_value = self.position * current_price
        total_capital = self.cash + position_value
        pnl = total_capital - self.capital
        
        return pnl

    def update_history(self, market_state: Dict) -> None:
        current_time = market_state["time_step"]
        true_price = market_state["true_price"]

        mid_price = (self.current_bid + self.current_ask) / 2
        spread = self.current_ask - self.current_bid

        pnl = self.calculate_pnl(true_price)
        total_capital = self.cash + (self.position * true_price)

        self.history["time"].append(current_time)
        self.history["position"].append(self.position)
        self.history["bid_price"].append(self.current_bid)
        self.history["ask_price"].append(self.current_ask)
        self.history["mid_price"].append(mid_price)
        self.history["spread"].append(spread)
        self.history["true_price"].append(true_price)
        self.history["capital"].append(total_capital)
        self.history["pnl"].append(pnl)
        
        self.history["var_95"].append(self.var_95)
        
        self.history["toxic_flow"].append(self.toxic_flow_metric)
        
        if self.use_ml_prediction:
            self.history["predicted_price"].append(self.price_prediction)
            self.history["price_error"].append(
                abs(self.price_prediction - true_price) / true_price if self.price_prediction > 0 else 0
            )
        else:
            self.history["predicted_price"].append(true_price)
            self.history["price_error"].append(0)
            
        self.history["market_regime"].append(market_state.get("market_regime", "normal"))