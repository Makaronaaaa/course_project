import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class RiskManager:
    def __init__(self, max_position: int = 100, max_drawdown_pct: float = 0.1,
                 var_limit_pct: float = 0.05, confidence_level: float = 0.95):
        self.max_position = max_position
        self.max_drawdown_pct = max_drawdown_pct
        self.var_limit_pct = var_limit_pct
        self.confidence_level = confidence_level
        
        self.position = 0
        self.position_value = 0
        self.highest_capital = 0
        self.current_capital = 0
        self.current_drawdown = 0
        self.var_95 = 0
        self.es_95 = 0
        self.position_concentration = 0
        
        self.drawdown_history = []
        self.var_history = []
        self.position_history = []
        
        self.inventory_cost = 0
        self.inventory_half_life = 60
    
    def update(self, position: int, position_value: float, price: float, 
               volatility: float, capital: float) -> Dict:
        self.position = position
        self.position_value = position_value
        
        if capital > self.highest_capital:
            self.highest_capital = capital
        
        self.current_capital = capital
        
        if self.highest_capital > 0:
            self.current_drawdown = (self.highest_capital - capital) / self.highest_capital
        
        self._calculate_var(position, position_value, price, volatility)
        
        self.position_concentration = abs(position) / self.max_position if self.max_position > 0 else 0
        
        self._calculate_inventory_cost(position)
        
        self.drawdown_history.append(self.current_drawdown)
        self.var_history.append(self.var_95)
        self.position_history.append(position)
        
        return {
            'var_95': self.var_95,
            'es_95': self.es_95,
            'drawdown': self.current_drawdown,
            'position_concentration': self.position_concentration,
            'inventory_cost': self.inventory_cost
        }
    
    def _calculate_var(self, position: int, position_value: float, price: float, volatility: float) -> None:
        daily_vol = volatility * math.sqrt(1/252)
        
        z_score = -1.645
        if self.confidence_level == 0.99:
            z_score = -2.326
        
        self.var_95 = abs(position_value) * daily_vol * abs(z_score)
        
        es_factor = 1.25
        self.es_95 = self.var_95 * es_factor
    
    def _calculate_inventory_cost(self, position: int) -> None:
        normalized_position = position / self.max_position if self.max_position > 0 else 0
        base_cost = normalized_position ** 2
        
        position_decay = 1.0
        
        self.inventory_cost = base_cost * position_decay
    
    def get_position_limit(self, price: float, volatility: float) -> int:
        base_limit = self.max_position
        
        drawdown_factor = 1.0 - self.current_drawdown / self.max_drawdown_pct
        drawdown_factor = max(0.2, min(1.0, drawdown_factor))
        
        var_pct = self.var_95 / self.current_capital if self.current_capital > 0 else 0
        var_factor = 1.0 - var_pct / self.var_limit_pct
        var_factor = max(0.2, min(1.0, var_factor))
        
        vol_factor = 0.01 / volatility if volatility > 0 else 1.0
        vol_factor = max(0.5, min(1.5, vol_factor))
        
        position_limit = int(base_limit * drawdown_factor * var_factor * vol_factor)
        
        return position_limit
    
    def is_safe_to_trade(self, order_size: int, price: float) -> bool:
        new_position = self.position + order_size
        if abs(new_position) > self.max_position:
            return False
        
        if self.current_drawdown >= self.max_drawdown_pct:
            return False
        
        if self.var_95 / self.current_capital >= self.var_limit_pct and self.current_capital > 0:
            if (self.position > 0 and order_size < 0) or (self.position < 0 and order_size > 0):
                return True
            return False
        
        return True
    
    def calculate_optimal_unwind(self, target_risk_reduction: float, price: float, 
                               market_impact: float) -> int:
        if self.position == 0:
            return 0
        
        direction = -1 if self.position > 0 else 1
        
        optimal_reduction = int(abs(self.position) * target_risk_reduction)
        
        optimal_reduction = max(1, min(optimal_reduction, abs(self.position)))
        
        return optimal_reduction * direction