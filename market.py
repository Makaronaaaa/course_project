import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

class Market:
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.01,
                 time_steps: int = 1000, informed_trader_ratio: float = 0.2,
                 fat_tails: bool = True, volatility_clustering: bool = True,
                 mean_reversion: float = 0.0, jump_intensity: float = 0.01,
                 data_from_csv: bool = False, csv_file: str = None):
        
        self.initial_price = initial_price
        self.volatility = volatility
        self.time_steps = time_steps
        self.informed_trader_ratio = informed_trader_ratio
        self.fat_tails = fat_tails
        self.volatility_clustering = volatility_clustering
        self.mean_reversion = mean_reversion
        self.jump_intensity = jump_intensity
        self.data_from_csv = data_from_csv
        self.csv_file = csv_file
        
        self.true_prices = self._generate_price_series()
        
        self.current_step = 0
        self.current_price = self.true_prices[0]
        self.trading_volume = 0
        self.volatility_estimate = volatility
        self.regime = "normal"

        self.session_open = True
        self.time_of_day = 0
        
        self.recent_trades = []
        self.recent_volumes = []
        self.order_imbalance = 0.0

    def _generate_price_series(self) -> np.ndarray:
        if self.data_from_csv and self.csv_file:
            """
            try:
                data = pd.read_csv(self.csv_file)
                if 'price' in data.columns:
                    # Ensure we have the right number of data points
                    prices = data['price'].values[:self.time_steps]
                    if len(prices) < self.time_steps:
                        # Pad with generated data if needed
                        print(f"Warning: CSV file has only {len(prices)} data points. Generating the rest.")
                        additional_prices = self._generate_synthetic_prices(self.time_steps - len(prices))
                        initial_value = prices[-1] if len(prices) > 0 else self.initial_price
                        additional_prices[0] = initial_value
                        for i in range(1, len(additional_prices)):
                            additional_prices[i] = additional_prices[i-1] * (1 + np.random.normal(0, self.volatility))
                        prices = np.concatenate([prices, additional_prices])
                    return prices
                else:
                    print("Error: CSV file does not contain a 'price' column. Generating synthetic data.")
            except Exception as e:
                print(f"Error loading CSV file: {e}. Generating synthetic data.")
            """
            pass
            
        return self._generate_synthetic_prices(self.time_steps)
    
    def _generate_synthetic_prices(self, length: int) -> np.ndarray:
        prices = np.zeros(length)
        prices[0] = self.initial_price
        
        if self.fat_tails:
            degrees_of_freedom = 5
            returns = np.random.standard_t(degrees_of_freedom, length) * self.volatility / np.sqrt(252)
            returns = returns * np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
        else:
            returns = np.random.normal(0, self.volatility / np.sqrt(252), length)

        if self.volatility_clustering:
            vol = np.ones(length) * self.volatility
            for i in range(1, length):
                vol[i] = np.sqrt(0.05 + 0.9 * vol[i-1]**2 + 0.05 * returns[i-1]**2)
                returns[i] = returns[i] * (vol[i] / self.volatility)
        
        for i in range(1, length):
            mean_reversion_component = self.mean_reversion * (np.log(self.initial_price) - np.log(prices[i-1]))
            
            jump = 0
            if np.random.random() < self.jump_intensity:
                jump_size = np.random.normal(0, self.volatility * 5)
                jump = jump_size
            
            price_change = returns[i] + mean_reversion_component + jump
            prices[i] = prices[i-1] * np.exp(price_change)
        
        return prices

    def step(self) -> Optional[Dict]:
        if self.current_step >= self.time_steps - 1:
            return None

        self.current_step += 1
        self.current_price = self.true_prices[self.current_step]
        
        self.time_of_day = (self.current_step % 390) / 390.0
        
        if self.current_step > 1:
            previous_price = self.true_prices[self.current_step - 1]
            return_t = np.log(self.current_price / previous_price)
            
            self.volatility_estimate = 0.94 * self.volatility_estimate + 0.06 * (return_t ** 2)
        
        self._update_market_regime()
        
        base_volume = self._generate_trading_volume()
        volatility_factor = 1 + 5 * self.volatility_estimate
        time_factor = self._time_of_day_factor()
        regime_factor = self._regime_volume_factor()
        
        self.trading_volume = int(base_volume * volatility_factor * time_factor * regime_factor)
        
        self.order_imbalance = self._generate_order_imbalance()
        
        self.recent_trades.append(self.current_price)
        self.recent_volumes.append(self.trading_volume)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
            self.recent_volumes.pop(0)
        
        return {
            "time_step": self.current_step,
            "true_price": self.current_price,
            "volatility": self.volatility_estimate,
            "trading_volume": self.trading_volume,
            "market_regime": self.regime,
            "time_of_day": self.time_of_day,
            "order_imbalance": self.order_imbalance
        }

    def _update_market_regime(self):
        if self.current_step < 5:
            return
        
        recent_returns = [np.log(self.true_prices[i] / self.true_prices[i-1])
                         for i in range(self.current_step-4, self.current_step+1)]
        recent_vol = np.std(recent_returns) * np.sqrt(252)
        
        if recent_vol > self.volatility * 3:
            self.regime = "crisis"
        elif recent_vol > self.volatility * 2:
            self.regime = "volatile"
        elif np.abs(np.mean(recent_returns) * 252) > self.volatility * 2:
            self.regime = "trending"
        else:
            self.regime = "normal"

    def _generate_trading_volume(self) -> int:
        r = 5
        p = 0.1
        return np.random.negative_binomial(r, p) + 10

    def _time_of_day_factor(self) -> float:
        t = self.time_of_day
        return 1.5 - 1.0 * np.sin(np.pi * t)

    def _regime_volume_factor(self) -> float:
        if self.regime == "crisis":
            return 2.5
        elif self.regime == "volatile":
            return 1.7
        elif self.regime == "trending":
            return 1.3
        else:
            return 1.0

    def _generate_order_imbalance(self) -> float:
        trend_bias = 0
        if self.current_step > 5:
            recent_return = (self.current_price / self.true_prices[self.current_step-5]) - 1
            trend_bias = np.clip(recent_return * 10, -0.5, 0.5)
        
        imbalance = np.random.normal(trend_bias, 0.2)
        return np.clip(imbalance, -1, 1)

    def reset(self):
        self.current_step = 0
        self.current_price = self.true_prices[0]
        self.trading_volume = 0
        self.volatility_estimate = self.volatility
        self.regime = "normal"
        self.recent_trades = []
        self.recent_volumes = []
        self.order_imbalance = 0.0