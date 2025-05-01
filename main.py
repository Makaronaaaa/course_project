import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from market import Market
from market_maker import MarketMaker
from traders import InformedTrader, LiquidityTrader
from visualization import plot_market_maker_results, compare_strategies

def run_simulation():
    market = Market(
        initial_price=100.0, 
        volatility=0.01, 
        time_steps=1000,
        fat_tails=True,
        volatility_clustering=True
    )

    market_maker = MarketMaker(
        "MainMM", 
        strategy_type="adaptive",
        use_ml_prediction=True,
        information_risk_management=True
    )
    
    informed_trader = InformedTrader(
        "InformedTrader", 
        information_advantage=0.2,
        realistic_behavior=True
    )
    
    liquidity_traders = [
        LiquidityTrader(f"LiquidityTrader_{i}", 
                         realistic_behavior=True) 
        for i in range(5)
    ]

    while True:
        market_state = market.step()
        if market_state is None:
            break

        market_state["future_prices"] = market.true_prices

        bid_price, ask_price = market_maker.calculate_quotes(market_state)

        order_size, price = informed_trader.place_order(bid_price, ask_price, market_state)
        if order_size != 0:
            market_maker.execute_trade(order_size, price)

        for trader in liquidity_traders:
            order_size, price = trader.place_order(bid_price, ask_price, market_state)
            if order_size != 0:
                market_maker.execute_trade(order_size, price)

        market_maker.update_history(market_state)

    print("Visualizations")

    plot_market_maker_results(market_maker, market)

    print("\nComparing different market maker strategies...")
    compare_strategies(market, ["passive", "aggressive", "adaptive", "ml_enhanced"])


if __name__ == "__main__":
    np.random.seed(42)
    run_simulation()