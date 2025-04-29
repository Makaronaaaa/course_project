import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import datetime as dt
import os
import requests
from io import StringIO
import json
from typing import Dict, List, Tuple, Union, Optional
import random

np.random.seed(42)
random.seed(42)

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class MarketEnvironment:
    def __init__(self, 
                 initial_price: float = 100.0, 
                 volatility: float = 0.01, 
                 steps: int = 1000,
                 data_source: str = 'generated'):
        self.initial_price = initial_price
        self.base_volatility = volatility
        self.steps = steps
        self.data_source = data_source
        self.current_step = 0
        
        self.prices = None
        self.volatility = None
        self.true_values = None
        self.current_price = None
        self.current_volatility = None
        self.time_indices = None

        self.reset()
    
    def reset(self):
        if self.data_source == 'exchange':
            try:
                self._load_exchange_data()
                print("Successfully loaded exchange data.")
            except Exception as e:
                print(f"Failed to load exchange data: {e}. Falling back to generated data.")
                self.data_source = 'generated'
                self._generate_synthetic_data()
        else:
            self._generate_synthetic_data()
            
        self.current_step = 0
        self.current_price = self.prices[0]
        self.current_volatility = self.volatility[0]
    
    def _load_exchange_data(self):
        try:
            ticker = "SBER"
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.csv"

            raise ConnectionError("Simulated failure to connect to exchange API")
            
            # If the above didn't raise an exception:
            # response = requests.get(url)
            # data = pd.read_csv(StringIO(response.text), sep=';')
            # self.prices = data['close'].values
            # self.volatility = self._calculate_volatility(self.prices)
            # self.steps = len(self.prices)
            # self.time_indices = pd.date_range(end=dt.datetime.now(), periods=self.steps, freq='1min')
            # self.true_values = self._generate_true_values()
            
        except Exception as e:
            print(f"Error loading exchange data: {e}")
            raise
    
    def _generate_synthetic_data(self):
        print("Generating synthetic market data...")
        
        self.time_indices = pd.date_range(
            end=dt.datetime.now(), 
            periods=self.steps, 
            freq='1min'
        )
        
        log_returns = np.random.normal(0.00005, self.base_volatility, self.steps)
        
        vol_cluster = 0.9
        for i in range(1, self.steps):
            log_returns[i] = (1 - vol_cluster) * log_returns[i] + vol_cluster * log_returns[i-1]
        
        self.prices = np.zeros(self.steps)
        self.prices[0] = self.initial_price
        for i in range(1, self.steps):
            self.prices[i] = self.prices[i-1] * np.exp(log_returns[i])
        
        self.volatility = np.zeros(self.steps)
        baseline_vol = self.base_volatility * self.prices
        
        garch_a = 0.1
        garch_b = 0.85
        self.volatility[0] = baseline_vol[0]
        
        for i in range(1, self.steps):
            self.volatility[i] = np.sqrt(0.05 * baseline_vol[i]**2 +
                                         garch_a * log_returns[i-1]**2 * self.prices[i-1]**2 + 
                                         garch_b * self.volatility[i-1]**2)
            
            if random.random() < 0.01:
                self.volatility[i] *= random.uniform(2.0, 5.0)
                
                shock_direction = 1 if random.random() > 0.5 else -1
                shock_magnitude = random.uniform(0.005, 0.02)
                self.prices[i] *= (1 + shock_direction * shock_magnitude)
        
        self._generate_true_values()
        
        print(f"Generated synthetic data with {self.steps} time steps.")
        print(f"Initial price: {self.prices[0]:.2f}, Final price: {self.prices[-1]:.2f}")
        print(f"Min volatility: {np.min(self.volatility):.4f}, Max volatility: {np.max(self.volatility):.4f}")

    def _generate_true_values(self):
        self.true_values = np.zeros(self.steps)
        look_ahead = min(5, self.steps // 100)

        for i in range(self.steps):
            future_idx = min(i + look_ahead, self.steps - 1)
            
            noise = np.random.normal(0, 0.001 * self.prices[i])
            self.true_values[i] = self.prices[future_idx] + noise
    
    def step(self):
        if self.current_step >= self.steps - 1:
            print("Simulation complete. Call reset() to start over.")
            return None
        
        self.current_step += 1
        self.current_price = self.prices[self.current_step]
        self.current_volatility = self.volatility[self.current_step]
        
        market_state = {
            'timestamp': self.time_indices[self.current_step],
            'price': self.current_price,
            'volatility': self.current_volatility,
            'true_value': self.true_values[self.current_step]  # Only informed traders use this
        }
        
        return market_state
    
    def get_current_state(self):
        return {
            'timestamp': self.time_indices[self.current_step],
            'price': self.current_price,
            'volatility': self.current_volatility,
            'true_value': self.true_values[self.current_step]
        }


class MarketMaker:
    def __init__(self, 
                 initial_capital: float = 1000000.0,
                 strategy_type: str = 'adaptive',
                 risk_aversion: float = 0.05,
                 info_parameter: float = 0.08,
                 max_position: int = 1000):

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy_type = strategy_type
        self.risk_aversion = risk_aversion
        self.info_parameter = info_parameter
        self.max_position = max_position
        
        self.position = 0
        self.cash = initial_capital
        self.fair_price_estimate = None
        
        self.bid_price = None
        self.ask_price = None
        self.bid_volume = 0
        self.ask_volume = 0
        
        self.capital_history = []
        self.position_history = []
        self.spread_history = []
        self.profit_history = []
        self.fair_price_history = []
        self.trades_executed = []
        
        self.alpha = 1.3
        self.order_intensity = 100.0
        self.optimization_horizon = 120
        
        self.strategy_probabilities = {'passive': 0.33, 'neutral': 0.34, 'aggressive': 0.33}
        self.update_strategy_probabilities()
        
        self.daily_profits = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_capital = initial_capital
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.cash = self.initial_capital
        self.fair_price_estimate = None
        
        self.capital_history = []
        self.position_history = []
        self.spread_history = []
        self.profit_history = []
        self.fair_price_history = []
        self.trades_executed = []
        
        self.daily_profits = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_capital = self.initial_capital
    
    def update_fair_price_estimate(self, market_state: dict, order_flow: list = None):
        current_price = market_state['price']
        
        if self.fair_price_estimate is None:
            self.fair_price_estimate = current_price
        
        self.fair_price_estimate = 0.9 * self.fair_price_estimate + 0.1 * current_price
        
        if order_flow:
            buy_volume = sum(order['volume'] for order in order_flow if order['type'] == 'buy')
            sell_volume = sum(order['volume'] for order in order_flow if order['type'] == 'sell')
            
            if buy_volume + sell_volume > 0:
                imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                
                adjustment = self.info_parameter * imbalance * current_price * 0.001
                self.fair_price_estimate += adjustment
        
        self.fair_price_history.append(self.fair_price_estimate)
    
    def set_quotes(self, market_state: dict, remaining_time: float = 1.0):
        current_price = market_state['price']
        current_volatility = market_state['volatility']
        
        self.update_fair_price_estimate(market_state)
        
        base_spread = current_volatility * np.sqrt(self.risk_aversion)
        
        if self.strategy_type == 'passive':
            spread_multiplier = 0.8
        elif self.strategy_type == 'aggressive':
            spread_multiplier = 1.5
        else:
            position_penalty = 0.1 * abs(self.position) / self.max_position
            vol_adjustment = (current_volatility / 0.01 - 1) * 0.2
            spread_multiplier = 1.0 + position_penalty + max(0, vol_adjustment)
        
        inventory_risk = self.risk_aversion * current_volatility * self.position * remaining_time
        
        half_spread = base_spread * spread_multiplier / 2
        
        self.bid_price = self.fair_price_estimate - half_spread - inventory_risk
        self.ask_price = self.fair_price_estimate + half_spread - inventory_risk
        
        min_spread = current_price * 0.0005
        if (self.ask_price - self.bid_price) < min_spread:
            half_min_spread = min_spread / 2
            self.bid_price = self.fair_price_estimate - half_min_spread - inventory_risk
            self.ask_price = self.fair_price_estimate + half_min_spread - inventory_risk
        
        max_volume = min(100, self.max_position - self.position)
        min_volume = min(100, self.max_position + self.position)
        
        if self.position > 0:
            self.ask_volume = max_volume
            self.bid_volume = max(10, min_volume / (1 + abs(self.position) / 200))
        elif self.position < 0:
            self.bid_volume = min_volume
            self.ask_volume = max(10, max_volume / (1 + abs(self.position) / 200))
        else:
            self.bid_volume = min_volume
            self.ask_volume = max_volume
        
        self.spread_history.append(self.ask_price - self.bid_price)
        
        return {
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume
        }
    
    def process_trade(self, trade: dict):
        trade_type = trade['type']  # 'buy' or 'sell'
        price = trade['price']
        volume = trade['volume']
        
        trade_record = {
            'timestamp': trade['timestamp'],
            'type': trade_type,
            'price': price,
            'volume': volume
        }
        
        self.trades_executed.append(trade_record)
        
        if trade_type == 'buy':
            self.position -= volume
            self.cash += price * volume
        else:
            self.position += volume
            self.cash -= price * volume
        
        if abs(self.position) > self.max_position:
            print(f"Warning: Position limit exceeded: {self.position}. Adjusting strategy.")
            self.risk_aversion *= 1.2
        
        market_value = self.position * price
        self.capital = self.cash + market_value
        
        self.capital_history.append(self.capital)
        self.position_history.append(self.position)
        
        trade_profit = 0
        if trade_type == 'buy':
            trade_profit = (price - self.fair_price_estimate) * volume
        else:
            trade_profit = (self.fair_price_estimate - price) * volume
        
        self.profit_history.append(trade_profit)
        
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def update_strategy_probabilities(self):
        payoff_matrix = {
            'mm_strategy': ['passive', 'neutral', 'aggressive'],
            'trader_strategy': ['aggressive', 'moderate', 'ignore'],
            'payoffs': [
                [(-5, 10), (0, 3), (2, 0)],
                [(-2, 7), (3, 2), (3, 0)],
                [(-8, 5), (1, 1), (1, 0)]
            ]
        }

        if len(self.trades_executed) > 20:
            recent_trades = self.trades_executed[-20:]
            aggressive_count = sum(1 for t in recent_trades if abs(t['volume']) > 50)
            moderate_count = sum(1 for t in recent_trades if 20 <= abs(t['volume']) <= 50)
            ignore_count = sum(1 for t in recent_trades if abs(t['volume']) < 20)
            
            total = len(recent_trades)
            if total > 0:
                trader_probs = {
                    'aggressive': aggressive_count / total,
                    'moderate': moderate_count / total,
                    'ignore': ignore_count / total
                }
            else:
                trader_probs = {'aggressive': 0.2, 'moderate': 0.5, 'ignore': 0.3}
        else:
            trader_probs = {'aggressive': 0.2, 'moderate': 0.5, 'ignore': 0.3}
        
        expected_payoffs = {}
        for i, mm_strategy in enumerate(payoff_matrix['mm_strategy']):
            payoff = 0
            for j, trader_strategy in enumerate(payoff_matrix['trader_strategy']):
                payoff += payoff_matrix['payoffs'][i][j][0] * trader_probs[trader_strategy]
            expected_payoffs[mm_strategy] = payoff
        
        best_strategy = max(expected_payoffs, key=expected_payoffs.get)
        
        epsilon = 0.1  # Exploration parameter
        total = sum(expected_payoffs.values())
        
        if total <= 0:
            adjusted_payoffs = {k: v + abs(min(0, min(expected_payoffs.values()))) + 1 
                               for k, v in expected_payoffs.items()}
            total = sum(adjusted_payoffs.values())
            expected_payoffs = adjusted_payoffs
        
        self.strategy_probabilities = {
            strategy: (1-epsilon) * (payoff / total) + epsilon/3
            for strategy, payoff in expected_payoffs.items()
        }
        
        strategies = list(self.strategy_probabilities.keys())
        probabilities = [self.strategy_probabilities[s] for s in strategies]
        self.strategy_type = np.random.choice(strategies, p=probabilities)
    
    def calculate_metrics(self, days=20):
        if len(self.capital_history) < 2:
            return {
                'avg_daily_profit': 0.0,
                'profit_volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        if len(self.capital_history) < days:
            days = 1
        
        periods = len(self.capital_history) // days
        if periods < 1:
            periods = 1
        
        daily_returns = []
        for i in range(days):
            start_idx = i * periods
            end_idx = min((i + 1) * periods, len(self.capital_history) - 1)
            if start_idx >= end_idx:
                break
            
            daily_return = (self.capital_history[end_idx] - self.capital_history[start_idx]) / self.capital_history[start_idx]
            daily_returns.append(daily_return)
        
        avg_daily_profit = np.mean(daily_returns) * self.initial_capital if daily_returns else 0
        profit_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        
        sharpe_ratio = (np.mean(daily_returns) / profit_volatility) * np.sqrt(252) if profit_volatility > 0 else 0
        
        return {
            'avg_daily_profit': avg_daily_profit,
            'profit_volatility': profit_volatility * 100,  # as percentage
            'max_drawdown': self.max_drawdown * 100,  # as percentage
            'sharpe_ratio': sharpe_ratio
        }


class InformedTrader:
    def __init__(self, 
                 initial_capital: float = 500000.0,
                 aggressiveness: float = 0.7,
                 info_accuracy: float = 0.8):
        self.capital = initial_capital
        self.aggressiveness = aggressiveness
        self.info_accuracy = info_accuracy
        self.position = 0
        
        self.strategies = ['aggressive', 'moderate', 'ignore']
        self.strategy_probabilities = {'aggressive': 0.3, 'moderate': 0.6, 'ignore': 0.1}
        self.current_strategy = np.random.choice(
            self.strategies, 
            p=[self.strategy_probabilities[s] for s in self.strategies]
        )
        
        self.mm_strategy_belief = {'passive': 0.33, 'neutral': 0.34, 'aggressive': 0.33}
    
    def decide_action(self, market_state: dict, quotes: dict):
        true_value = market_state['true_value']
        current_price = market_state['price']
        bid_price = quotes['bid_price']
        ask_price = quotes['ask_price']
        bid_volume = quotes['bid_volume']
        ask_volume = quotes['ask_volume']
        
        expected_value = self.info_accuracy * true_value + (1 - self.info_accuracy) * current_price
        
        buy_profit = expected_value - ask_price
        sell_profit = bid_price - expected_value
        
        self.update_strategy(quotes)
        
        if self.current_strategy == 'aggressive':
            min_profit_threshold = 0.0001 * current_price
        elif self.current_strategy == 'moderate':
            min_profit_threshold = 0.001 * current_price
        else:
            return None
        
        if buy_profit > min_profit_threshold:
            size = min(
                int(self.aggressiveness * 100 * (buy_profit / current_price) * 1000),
                ask_volume,
                int(self.capital / ask_price / 2)
            )
            if size > 0:
                return {
                    'type': 'buy',
                    'price': ask_price,
                    'volume': size,
                    'timestamp': market_state['timestamp']
                }
        elif sell_profit > min_profit_threshold:
            size = min(
                int(self.aggressiveness * 100 * (sell_profit / current_price) * 1000),
                bid_volume,
                abs(self.position)
            )
            if size > 0:
                return {
                    'type': 'sell',
                    'price': bid_price,
                    'volume': size,
                    'timestamp': market_state['timestamp']
                }
        
        return None
    
    def update_strategy(self, quotes: dict):
        spread = quotes['ask_price'] - quotes['bid_price']
        mid_price = (quotes['ask_price'] + quotes['bid_price']) / 2
        relative_spread = spread / mid_price
        
        if relative_spread < 0.001:
            mm_observed = 'passive'
        elif relative_spread > 0.002:
            mm_observed = 'aggressive'
        else:
            mm_observed = 'neutral'

        alpha = 0.2
        for strat in self.mm_strategy_belief:
            if strat == mm_observed:
                self.mm_strategy_belief[strat] = (1-alpha) * self.mm_strategy_belief[strat] + alpha
            else:
                self.mm_strategy_belief[strat] = (1-alpha) * self.mm_strategy_belief[strat]
        
        total = sum(self.mm_strategy_belief.values())
        self.mm_strategy_belief = {k: v/total for k, v in self.mm_strategy_belief.items()}
        
        payoff_matrix = {
            'trader_strategy': ['aggressive', 'moderate', 'ignore'],
            'mm_strategy': ['passive', 'neutral', 'aggressive'],
            'payoffs': [
                [(10, -5), (7, -2), (5, -8)],
                [(3, 0), (2, 3), (1, 1)],
                [(0, 2), (0, 3), (0, 1)]
            ]
        }
        
        expected_payoffs = {}
        for i, trader_strategy in enumerate(payoff_matrix['trader_strategy']):
            payoff = 0
            for j, mm_strategy in enumerate(payoff_matrix['mm_strategy']):
                payoff += payoff_matrix['payoffs'][i][j][0] * self.mm_strategy_belief[mm_strategy]
            expected_payoffs[trader_strategy] = payoff
        
        epsilon = 0.1
        best_strategy = max(expected_payoffs, key=expected_payoffs.get)
        
        if random.random() < epsilon:
            self.current_strategy = random.choice(self.strategies)
        else:
            self.current_strategy = best_strategy
    
    def process_trade_result(self, trade: dict, current_price: float):
        if trade['type'] == 'buy':
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        else:  # sell
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']


class LiquidityTrader:
    def __init__(self, trade_frequency: float = 0.2):
        self.trade_frequency = trade_frequency
    
    def decide_action(self, market_state: dict, quotes: dict):
        if random.random() > self.trade_frequency:
            return None
        
        buy = random.random() > 0.5
        
        bid_price = quotes['bid_price']
        ask_price = quotes['ask_price']
        bid_volume = quotes['bid_volume']
        ask_volume = quotes['ask_volume']
        
        if buy:
            size = random.randint(1, min(50, ask_volume))
            return {
                'type': 'buy',
                'price': ask_price,
                'volume': size,
                'timestamp': market_state['timestamp']
            }
        else:
            size = random.randint(1, min(50, bid_volume))
            return {
                'type': 'sell',
                'price': bid_price,
                'volume': size,
                'timestamp': market_state['timestamp']
            }


class MarketSimulator:
    def __init__(self, 
                 market_env: MarketEnvironment,
                 strategies: List[str] = ['passive', 'aggressive', 'adaptive']):

        self.market_env = market_env
        self.strategies = strategies
        
        self.market_makers = {
            strategy: MarketMaker(strategy_type=strategy) 
            for strategy in strategies
        }
        
        self.informed_trader = InformedTrader()
        self.liquidity_trader = LiquidityTrader()
        
        self.results = {strategy: {} for strategy in strategies}
    
    def reset(self):
        self.market_env.reset()
        for mm in self.market_makers.values():
            mm.reset()
        self.informed_trader = InformedTrader()
        self.liquidity_trader = LiquidityTrader()
    
    def run_simulation(self):
        print("Starting market simulation...")
        self.reset()
        
        total_steps = self.market_env.steps
        
        for step in range(total_steps):
            if step % 100 == 0:
                print(f"Simulation progress: {step}/{total_steps} steps ({step/total_steps*100:.1f}%)")
            
            market_state = self.market_env.step()
            if market_state is None:
                break
            
            remaining_time = 1.0 - (step / total_steps)
            quotes = {
                strategy: mm.set_quotes(market_state, remaining_time)
                for strategy, mm in self.market_makers.items()
            }
            
            for strategy, mm_quotes in quotes.items():
                informed_trade = self.informed_trader.decide_action(market_state, mm_quotes)
                if informed_trade:
                    self.market_makers[strategy].process_trade(informed_trade)
                    self.informed_trader.process_trade_result(informed_trade, market_state['price'])
            
            for strategy, mm_quotes in quotes.items():
                liquidity_trade = self.liquidity_trader.decide_action(market_state, mm_quotes)
                if liquidity_trade:
                    self.market_makers[strategy].process_trade(liquidity_trade)
            
            if step % 20 == 0:
                for mm in self.market_makers.values():
                    mm.update_strategy_probabilities()
                self.informed_trader.update_strategy(
                    quotes[random.choice(list(quotes.keys()))]
                )
        
        print("Simulation complete!")
        
        for strategy, mm in self.market_makers.items():
            self.results[strategy] = mm.calculate_metrics()
            print(f"Strategy: {strategy}")
            for metric, value in self.results[strategy].items():
                print(f"  {metric}: {value:.2f}")
        
        return self.results
    
    def plot_results(self):
        if not all(mm.capital_history for mm in self.market_makers.values()):
            print("No simulation data to plot. Run simulation first.")
            return
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        self._plot_price_and_spread()
        self._plot_profit_comparison()
        self._plot_position_comparison()
        self._plot_strategy_metrics()
        self._plot_spread_distribution()
        
        print("Plots saved to 'plots' directory.")
    
    def _plot_price_and_spread(self):
        time_indices = self.market_env.time_indices[:len(self.market_makers['adaptive'].spread_history)]
        prices = self.market_env.prices[:len(time_indices)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        ax1.plot(time_indices, prices, label='Asset Price', color='black', linewidth=2)
        ax1.set_ylabel('Price')
        ax1.set_title('Asset Price Over Time')
        ax1.grid(True)
        ax1.legend()
        
        for strategy, mm in self.market_makers.items():
            spreads = mm.spread_history
            if len(spreads) > len(time_indices):
                spreads = spreads[:len(time_indices)]
            elif len(spreads) < len(time_indices):
                time_indices_adj = time_indices[:len(spreads)]
            else:
                time_indices_adj = time_indices
            
            relative_spreads = [s / p for s, p in zip(spreads, prices[:len(spreads)])]
            ax2.plot(time_indices_adj, relative_spreads, label=f'{strategy.capitalize()} Strategy')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Relative Spread')
        ax2.set_title('Bid-Ask Spread by Strategy')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('plots/price_and_spread.png', dpi=300)
        plt.close()
    
    def _plot_profit_comparison(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for strategy, mm in self.market_makers.items():
            capital_history = mm.capital_history
            initial_capital = mm.initial_capital
            
            profit_pct = [(c - initial_capital) / initial_capital * 100 for c in capital_history]
            
            time_indices = self.market_env.time_indices[:len(profit_pct)]
            
            ax.plot(time_indices, profit_pct, label=f'{strategy.capitalize()} Strategy', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Profit (%)')
        ax.set_title('Market Maker Profit Comparison')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/profit_comparison.png', dpi=300)
        plt.close()
    
    def _plot_position_comparison(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for strategy, mm in self.market_makers.items():
            position_history = mm.position_history
            
            time_indices = self.market_env.time_indices[:len(position_history)]
            
            ax.plot(time_indices, position_history, label=f'{strategy.capitalize()} Strategy', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Position (shares)')
        ax.set_title('Market Maker Position Comparison')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/position_comparison.png', dpi=300)
        plt.close()
    
    def _plot_strategy_metrics(self):
        strategies = list(self.strategies)
        
        metrics = ['avg_daily_profit', 'profit_volatility', 'max_drawdown', 'sharpe_ratio']
        metric_data = {
            metric: [self.results[strategy][metric] for strategy in strategies]
            for metric in metrics
        }
        
        metric_labels = {
            'avg_daily_profit': 'Avg. Daily Profit',
            'profit_volatility': 'Profit Volatility (%)',
            'max_drawdown': 'Max Drawdown (%)',
            'sharpe_ratio': 'Sharpe Ratio'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(strategies, metric_data[metric])
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax.set_xlabel('Strategy')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(metric_labels[metric])
            ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('plots/strategy_metrics.png', dpi=300)
        plt.close()
    
    def _plot_spread_distribution(self):
        fig, axes = plt.subplots(len(self.strategies), 1, figsize=(14, 4 * len(self.strategies)))
        
        if len(self.strategies) == 1:
            axes = [axes]
        
        prices = self.market_env.prices
        
        for i, strategy in enumerate(self.strategies):
            mm = self.market_makers[strategy]
            spreads = mm.spread_history
            
            relative_spreads = [spread / price * 10000 for spread, price in zip(spreads, prices[:len(spreads)])]
            
            axes[i].hist(relative_spreads, bins=50, alpha=0.7)
            axes[i].set_xlabel('Spread (basis points)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of Spreads - {strategy.capitalize()} Strategy')
            axes[i].grid(True)
            
            mean_spread = np.mean(relative_spreads)
            median_spread = np.median(relative_spreads)
            
            axes[i].axvline(mean_spread, color='r', linestyle='--', label=f'Mean: {mean_spread:.2f} bp')
            axes[i].axvline(median_spread, color='g', linestyle='-.', label=f'Median: {median_spread:.2f} bp')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('plots/spread_distribution.png', dpi=300)
        plt.close()


def run_market_maker_simulation():
    print("Market Maker Simulation with Game Theory")
    print("----------------------------------------")
    
    market_env = MarketEnvironment(
        initial_price=100.0,
        volatility=0.01,
        steps=1000,
        data_source='generated'  # Start with generated data
    )
    
    simulator = MarketSimulator(
        market_env=market_env,
        strategies=['passive', 'aggressive', 'adaptive']
    )
    
    results = simulator.run_simulation()
    
    simulator.plot_results()
    
    print("\nPerformance Metrics Comparison:")
    print("-------------------------------")
    
    metrics = ['avg_daily_profit', 'profit_volatility', 'max_drawdown', 'sharpe_ratio']
    metric_labels = {
        'avg_daily_profit': 'Avg. Daily Profit',
        'profit_volatility': 'Profit Volatility (%)',
        'max_drawdown': 'Max Drawdown (%)',
        'sharpe_ratio': 'Sharpe Ratio'
    }
    
    header = "| Strategy | " + " | ".join(metric_labels.values()) + " |"
    separator = "|" + "-" * (len(header) - 2) + "|"
    
    print(separator)
    print(header)
    print(separator)
    
    for strategy in simulator.strategies:
        row = f"| {strategy.capitalize()} | "
        for metric in metrics:
            row += f"{results[strategy][metric]:.2f} | "
        print(row)
    
    print(separator)
    
    print("\nSimulation complete. Results are available in the 'plots' directory.")
    return results


if __name__ == "__main__":
    run_market_maker_simulation()