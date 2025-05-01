import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict
import seaborn as sns

def plot_market_maker_results(market_maker, market, figsize: Tuple[int, int] = (15, 12)):
    history = market_maker.history

    df = pd.DataFrame({
        'time': history['time'],
        'position': history['position'],
        'bid_price': history['bid_price'],
        'ask_price': history['ask_price'],
        'mid_price': history['mid_price'],
        'spread': history['spread'],
        'true_price': history['true_price'],
        'capital': history['capital'],
        'pnl': history['pnl'],
        'var_95': history.get('var_95', [0] * len(history['time'])),
        'toxic_flow': history.get('toxic_flow', [0] * len(history['time'])),
        'predicted_price': history.get('predicted_price', [0] * len(history['time'])),
        'price_error': history.get('price_error', [0] * len(history['time'])),
        'market_regime': history.get('market_regime', ['normal'] * len(history['time']))
    })

    df['spread_pct'] = df['spread'] / df['true_price'] * 100
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(8, 6, figure=fig)

    # Plot 1: Price and quotes
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(df['time'], df['true_price'], label='True Price', color='black', linewidth=1)
    ax1.plot(df['time'], df['bid_price'], label='Bid Price', color='green', alpha=0.7)
    ax1.plot(df['time'], df['ask_price'], label='Ask Price', color='red', alpha=0.7)
    
    if market_maker.use_ml_prediction:
        ax1.plot(df['time'], df['predicted_price'], label='Predicted Price', 
                 color='blue', linestyle='--', alpha=0.5)
    
    ax1.set_title('Asset Price and Market Maker Quotes')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Market Maker Position
    ax2 = fig.add_subplot(gs[2:4, :3])
    ax2.plot(df['time'], df['position'], label='Position', color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    max_pos = market_maker.max_position
    ax2.axhline(y=max_pos, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-max_pos, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Market Maker Position')
    ax2.set_ylabel('Position Size')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spread
    ax3 = fig.add_subplot(gs[2:4, 3:])
    ax3.plot(df['time'], df['spread_pct'], label='Spread %', color='purple')
    
    if 'market_regime' in df.columns and df['market_regime'].nunique() > 1:
        regime_colors = {
            'normal': 'white',
            'volatile': 'lightyellow',
            'trending': 'lightblue',
            'crisis': 'lightpink'
        }
        
        for regime in regime_colors.keys():
            if regime in df['market_regime'].values:
                mask = df['market_regime'] == regime
                ax3.fill_between(df['time'], 0, df['spread_pct'].max(), 
                                where=mask, color=regime_colors[regime], alpha=0.3,
                                label=f'{regime} regime')
    
    ax3.set_title('Bid-Ask Spread (% of Price)')
    ax3.set_ylabel('Spread %')
    ax3.grid(True, alpha=0.3)
    if 'market_regime' in df.columns and df['market_regime'].nunique() > 1:
        ax3.legend(loc='upper right')

    # Plot 4: P&L
    ax4 = fig.add_subplot(gs[4:6, :])
    ax4.plot(df['time'], df['pnl'], label='P&L', color='orange')
    ax4.fill_between(df['time'], 0, df['pnl'], where=df['pnl'] >= 0, color='green', alpha=0.3)
    ax4.fill_between(df['time'], 0, df['pnl'], where=df['pnl'] < 0, color='red', alpha=0.3)
    
    if 'var_95' in df.columns and df['var_95'].sum() > 0:
        ax4.plot(df['time'], -df['var_95'], label='95% VaR', color='red', linestyle='--', alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Market Maker Profit and Loss')
    ax4.set_ylabel('P&L')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Spread vs. Volatility
    ax5 = fig.add_subplot(gs[6:8, :3])
    volatility_series = [np.log(market.true_prices[i+1]/market.true_prices[i]) for i in range(len(market.true_prices)-1)]
    volatility_rolling = pd.Series(volatility_series).rolling(window=20).std() * np.sqrt(252)
    
    spread_vs_vol = pd.DataFrame({
        'spread_pct': df['spread_pct'],
        'rolling_vol': volatility_rolling
    }).dropna()

    hb = ax5.hexbin(spread_vs_vol['rolling_vol']*100, spread_vs_vol['spread_pct'],
                   gridsize=20, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=ax5, label='Count')
    
    if len(spread_vs_vol) > 0:
        try:
            z = np.polyfit(spread_vs_vol['rolling_vol']*100, spread_vs_vol['spread_pct'], 1)
            p = np.poly1d(z)
            ax5.plot(spread_vs_vol['rolling_vol']*100, p(spread_vs_vol['rolling_vol']*100), 
                    "r--", linewidth=1)
        except:
            pass
    
    ax5.set_title('Spread vs. Volatility')
    ax5.set_xlabel('20-period Rolling Volatility %')
    ax5.set_ylabel('Spread %')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Information Risk Management metrics (if available)
    ax6 = fig.add_subplot(gs[6:8, 3:])
    
    if market_maker.information_risk_management and 'toxic_flow' in df.columns:
        ax6.plot(df['time'], df['toxic_flow'], label='Toxic Flow Metric', color='red')
        ax6.set_title('Information Risk Metrics')
        ax6.set_ylabel('Toxic Flow')
        ax6.grid(True, alpha=0.3)
    elif market_maker.use_ml_prediction and 'price_error' in df.columns:
        ax6.plot(df['time'], df['price_error']*100, label='Prediction Error %', color='purple')
        ax6.set_title('ML Prediction Accuracy')
        ax6.set_ylabel('Prediction Error %')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.hist(df['spread_pct'], bins=30, alpha=0.7, color='green')
        ax6.set_title('Spread Distribution')
        ax6.set_xlabel('Spread %')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nMarket Maker Summary Statistics:")
    print(f"Strategy Type: {market_maker.strategy_type}")
    print(f"Final P&L: ${df['pnl'].iloc[-1]:.2f}")
    print(f"Final Position: {df['position'].iloc[-1]}")
    print(f"Average Spread: {df['spread_pct'].mean():.4f}%")
    print(f"Average Daily P&L: ${df['pnl'].iloc[-1]/market.time_steps*252:.2f}")
    print(f"Sharpe Ratio: {df['pnl'].diff().mean() / df['pnl'].diff().std() * np.sqrt(252) if df['pnl'].diff().std() > 0 else 0:.2f}")
    print(f"Maximum Drawdown: ${min(0, df['pnl'].min()):.2f}")
    print(f"Total Trades Executed: {market_maker.trades_executed}")
    
    if market_maker.trades_executed > 0:
        print(f"Average Trade Size: {sum(market_maker.trade_sizes)/len(market_maker.trade_sizes):.2f}")
    
    if market_maker.information_risk_management:
        print(f"\nInformation Risk Management:")
        print(f"Average Toxic Flow Metric: {df['toxic_flow'].mean():.4f}")
    
    if market_maker.use_ml_prediction:
        print(f"\nML Model Performance:")
        print(f"Average Prediction Error: {df['price_error'].mean()*100:.4f}%")


def compare_strategies(market: object, strategies: List[str],
                       figsize: Tuple[int, int] = (12, 12)) -> None:
    market.reset()

    market_makers = []
    for strategy in strategies:
        mm = MarketMaker(
            f"MM_{strategy}", 
            strategy_type=strategy,
            use_ml_prediction=(strategy == "ml_enhanced"),
            information_risk_management=(strategy in ["adaptive", "ml_enhanced"])
        )
        market_makers.append(mm)

    informed_trader = InformedTrader("InformedTrader", realistic_behavior=True)
    liquidity_traders = [LiquidityTrader(f"LiquidityTrader_{i}", realistic_behavior=True) 
                         for i in range(5)]

    while True:
        market_state = market.step()
        if market_state is None:
            break

        market_state["future_prices"] = market.true_prices

        for mm in market_makers:
            bid_price, ask_price = mm.calculate_quotes(market_state)

            order_size, price = informed_trader.place_order(bid_price, ask_price, market_state)
            if order_size != 0:
                mm.execute_trade(order_size, price)

            for trader in liquidity_traders:
                order_size, price = trader.place_order(bid_price, ask_price, market_state)
                if order_size != 0:
                    mm.execute_trade(order_size, price)

            mm.update_history(market_state)

    plt.figure(figsize=figsize)

    # Plot 1: P&L Comparison
    plt.subplot(3, 1, 1)
    for mm in market_makers:
        plt.plot(mm.history['time'], mm.history['pnl'], label=f"{mm.strategy_type}")
    plt.title('P&L Comparison of Different Strategies')
    plt.ylabel('P&L')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Spread Comparison
    plt.subplot(3, 1, 2)
    for mm in market_makers:
        spread_pct = [s/p*100 for s, p in zip(mm.history['spread'], mm.history['true_price'])]
        plt.plot(mm.history['time'], spread_pct, label=f"{mm.strategy_type}")
    plt.title('Spread Comparison of Different Strategies')
    plt.ylabel('Spread %')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Position Comparison
    plt.subplot(3, 1, 3)
    for mm in market_makers:
        plt.plot(mm.history['time'], mm.history['position'], label=f"{mm.strategy_type}")
    plt.title('Position Comparison of Different Strategies')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nStrategy Comparison:")
    print("-" * 85)
    print(f"{'Strategy':<15} {'Final P&L':<15} {'Avg Spread %':<15} {'Trades':<10} {'Sharpe':<10} {'Max DD':<10} {'Win %':<10}")
    print("-" * 85)

    for mm in market_makers:
        pnl = mm.history['pnl'][-1]
        spread_pct = [s/p*100 for s, p in zip(mm.history['spread'], mm.history['true_price'])]
        avg_spread = sum(spread_pct) / len(spread_pct)
        trades = mm.trades_executed

        pnl_diff = np.diff([0] + mm.history['pnl'])
        sharpe = pnl_diff.mean() / pnl_diff.std() * np.sqrt(252) if pnl_diff.std() > 0 else 0

        cum_max = np.maximum.accumulate(mm.history['pnl'])
        drawdown = cum_max - mm.history['pnl']
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        win_pct = sum(1 for d in pnl_diff if d > 0) / len(pnl_diff) * 100 if len(pnl_diff) > 0 else 0

        print(f"{mm.strategy_type:<15} ${pnl:<14.2f} {avg_spread:<14.4f} {trades:<10} {sharpe:<10.2f} ${max_dd:<9.2f} {win_pct:<10.1f}%")
    
    _plot_strategy_regime_heatmap(market_makers)


def _plot_strategy_regime_heatmap(market_makers):
    if 'market_regime' not in market_makers[0].history:
        return
    
    regime_performance = {}
    
    for mm in market_makers:
        strategy = mm.strategy_type
        regime_performance[strategy] = {}
        
        df = pd.DataFrame({
            'pnl': mm.history['pnl'],
            'pnl_diff': np.diff([0] + mm.history['pnl']),
            'regime': mm.history['market_regime']
        })
        
        for regime in df['regime'].unique():
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                avg_return = regime_data['pnl_diff'].mean()
                regime_performance[strategy][regime] = avg_return
    
    regimes = list(set().union(*[set(perf.keys()) for perf in regime_performance.values()]))
    strategies = list(regime_performance.keys())
    
    heatmap_data = np.zeros((len(strategies), len(regimes)))
    
    for i, strategy in enumerate(strategies):
        for j, regime in enumerate(regimes):
            if regime in regime_performance[strategy]:
                heatmap_data[i, j] = regime_performance[strategy][regime]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=regimes, yticklabels=strategies)
    plt.title("Average Return by Strategy and Market Regime")
    plt.tight_layout()
    plt.show()