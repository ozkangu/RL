"""
Evaluation Script - Faz 4: Değerlendirme ve Backtest
Evaluates trained RL agent, benchmarks, and generates comprehensive reports.

Epic 4.1: Evaluate Script (PBI-026, PBI-027)
Epic 4.2: Benchmarks (PBI-028, PBI-029)
Epic 4.3: Metrics (PBI-030, PBI-031, PBI-032)
Epic 4.4: Visualization (PBI-033, PBI-034, PBI-035, PBI-036)
"""

import argparse
import yaml
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data_manager import prepare_data_pipeline
from trading_env import BtcUsdTradingEnv
from config_validator import validate_all_configs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_model(model_path: str) -> PPO:
    """
    Load trained model from checkpoint.

    PBI-026: Load trained RL model.

    Args:
        model_path: Path to model checkpoint (.zip file)

    Returns:
        Loaded PPO model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    logger.info("Model loaded successfully")

    return model


def run_episode_rl_agent(env, model, deterministic: bool = True) -> Dict[str, Any]:
    """
    Run one episode with RL agent and collect detailed trade data.

    PBI-027: Episode rollout with trade blotter.

    Args:
        env: Trading environment
        model: Trained RL model
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary with episode data including trade blotter
    """
    logger.info("Running RL agent episode...")

    obs = env.reset()
    done = False

    # Trade blotter: detailed trade history
    trade_blotter = []
    portfolio_values = []
    actions_taken = []
    prices = []
    timestamps = []

    step = 0

    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)

        # Execute action
        obs, reward, done, info = env.step(action)

        # Extract info from environment
        env_info = info[0] if isinstance(info, list) else info

        # Record trade if transaction occurred
        if env_info.get('transaction_occurred', False):
            trade_entry = {
                'step': step,
                'timestamp': step,  # Will be replaced with actual timestamp if available
                'action': ['Hold', 'Buy', 'Sell'][int(action[0])],
                'price': env_info['current_price'],
                'position': env_info['position'],
                'portfolio_value': env_info['portfolio_value'],
                'cash': env_info['cash']
            }
            trade_blotter.append(trade_entry)

        # Record portfolio value and actions
        portfolio_values.append(env_info['portfolio_value'])
        actions_taken.append(int(action[0]))
        prices.append(env_info['current_price'])

        step += 1

    logger.info(f"Episode completed: {step} steps, {len(trade_blotter)} trades")

    return {
        'trade_blotter': pd.DataFrame(trade_blotter),
        'portfolio_values': np.array(portfolio_values),
        'actions': np.array(actions_taken),
        'prices': np.array(prices),
        'total_steps': step
    }


def buy_and_hold_benchmark(df: pd.DataFrame, initial_cash: float = 10000.0,
                           transaction_cost: float = 0.001) -> Dict[str, Any]:
    """
    Buy & Hold benchmark strategy.

    PBI-028: Simple buy at start, hold until end, then sell.

    Args:
        df: Market data DataFrame
        initial_cash: Starting capital
        transaction_cost: Transaction cost percentage

    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running Buy & Hold benchmark...")

    # Buy at first price
    first_price = df['close'].iloc[0]
    buy_cost = initial_cash * transaction_cost
    capital_after_buy = initial_cash - buy_cost
    btc_amount = capital_after_buy / first_price

    # Track portfolio value over time
    portfolio_values = []
    for price in df['close']:
        portfolio_values.append(btc_amount * price)

    # Sell at last price
    last_price = df['close'].iloc[-1]
    final_value_before_cost = btc_amount * last_price
    sell_cost = final_value_before_cost * transaction_cost
    final_value = final_value_before_cost - sell_cost

    # Trade blotter
    trade_blotter = pd.DataFrame([
        {
            'step': 0,
            'timestamp': 0,
            'action': 'Buy',
            'price': first_price,
            'position': 1,
            'portfolio_value': capital_after_buy,
            'cash': 0
        },
        {
            'step': len(df) - 1,
            'timestamp': len(df) - 1,
            'action': 'Sell',
            'price': last_price,
            'position': 0,
            'portfolio_value': final_value,
            'cash': final_value
        }
    ])

    logger.info(f"Buy & Hold: Initial=${initial_cash:.2f}, Final=${final_value:.2f}")

    return {
        'trade_blotter': trade_blotter,
        'portfolio_values': np.array(portfolio_values),
        'actions': np.ones(len(df), dtype=int),  # Always hold
        'prices': df['close'].values,
        'total_steps': len(df)
    }


def rsi_baseline_strategy(df: pd.DataFrame, initial_cash: float = 10000.0,
                         transaction_cost: float = 0.001,
                         rsi_low: int = 30, rsi_high: int = 70) -> Dict[str, Any]:
    """
    RSI baseline strategy.

    PBI-029: Buy when RSI < 30, Sell when RSI > 70.

    Args:
        df: Market data DataFrame (must have 'rsi_14' column)
        initial_cash: Starting capital
        transaction_cost: Transaction cost percentage
        rsi_low: RSI threshold for buying
        rsi_high: RSI threshold for selling

    Returns:
        Dictionary with strategy results
    """
    logger.info(f"Running RSI baseline strategy (RSI {rsi_low}/{rsi_high})...")

    if 'rsi_14' not in df.columns:
        logger.warning("RSI indicator not found in DataFrame. Using default RSI calculation.")
        # Simple RSI calculation if not present
        import ta
        df = df.copy()
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    trade_blotter = []
    portfolio_values = []
    actions_taken = []

    cash = initial_cash
    position = 0  # 0: no position, 1: long
    position_price = 0.0
    btc_amount = 0.0

    for idx, row in df.iterrows():
        price = row['close']
        rsi = row['rsi_14']

        action = 0  # Hold
        transaction_occurred = False

        # Buy signal: RSI < rsi_low and no position
        if rsi < rsi_low and position == 0 and not pd.isna(rsi):
            # Buy
            cost = cash * transaction_cost
            capital_for_btc = cash - cost
            btc_amount = capital_for_btc / price
            position = 1
            position_price = price
            cash = 0
            action = 1
            transaction_occurred = True

            trade_blotter.append({
                'step': len(portfolio_values),
                'timestamp': idx,
                'action': 'Buy',
                'price': price,
                'position': position,
                'portfolio_value': btc_amount * price,
                'cash': cash
            })

        # Sell signal: RSI > rsi_high and has position
        elif rsi > rsi_high and position == 1 and not pd.isna(rsi):
            # Sell
            cash_from_sale = btc_amount * price
            cost = cash_from_sale * transaction_cost
            cash = cash_from_sale - cost
            position = 0
            btc_amount = 0.0
            action = 2
            transaction_occurred = True

            trade_blotter.append({
                'step': len(portfolio_values),
                'timestamp': idx,
                'action': 'Sell',
                'price': price,
                'position': position,
                'portfolio_value': cash,
                'cash': cash
            })

        # Calculate portfolio value
        if position == 1:
            portfolio_value = btc_amount * price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)
        actions_taken.append(action)

    logger.info(f"RSI Strategy: {len(trade_blotter)} trades executed")

    return {
        'trade_blotter': pd.DataFrame(trade_blotter),
        'portfolio_values': np.array(portfolio_values),
        'actions': np.array(actions_taken),
        'prices': df['close'].values,
        'total_steps': len(df)
    }


def calculate_metrics(portfolio_values: np.ndarray, initial_cash: float = 10000.0,
                     trade_blotter: Optional[pd.DataFrame] = None,
                     risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    PBI-030: Basic metrics (Total Return, Sharpe Ratio, Max Drawdown, Win Rate)
    PBI-031: Advanced metrics (Sortino, Calmar, Avg Trade P&L, Turnover)

    Args:
        portfolio_values: Array of portfolio values over time
        initial_cash: Initial capital
        trade_blotter: DataFrame with trade history (for trade-based metrics)
        risk_free_rate: Annual risk-free rate (default: 2%)

    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}

    # PBI-030: Basic Metrics

    # Total Return
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_cash) / initial_cash
    metrics['total_return'] = total_return
    metrics['total_return_pct'] = total_return * 100

    # Returns series
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN values

    # Sharpe Ratio (assuming daily returns, annualized)
    if len(returns) > 0 and np.std(returns) > 0:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        # Annualize (assuming hourly data: 24*365 = 8760 hours per year)
        sharpe_ratio = (mean_return * 8760 - risk_free_rate) / (std_return * np.sqrt(8760))
        metrics['sharpe_ratio'] = sharpe_ratio
    else:
        metrics['sharpe_ratio'] = 0.0

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    metrics['max_drawdown'] = max_drawdown
    metrics['max_drawdown_pct'] = max_drawdown * 100

    # Win Rate (from trade blotter if available)
    if trade_blotter is not None and len(trade_blotter) > 0:
        # Calculate P&L for each trade pair (buy-sell)
        trades = trade_blotter[trade_blotter['action'].isin(['Buy', 'Sell'])].copy()

        if len(trades) >= 2:
            winning_trades = 0
            total_trades = 0
            trade_pnls = []

            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_trade = trades.iloc[i]
                    sell_trade = trades.iloc[i + 1]

                    if buy_trade['action'] == 'Buy' and sell_trade['action'] == 'Sell':
                        buy_price = buy_trade['price']
                        sell_price = sell_trade['price']
                        pnl = (sell_price - buy_price) / buy_price
                        trade_pnls.append(pnl)

                        if pnl > 0:
                            winning_trades += 1
                        total_trades += 1

            if total_trades > 0:
                metrics['win_rate'] = winning_trades / total_trades
                metrics['win_rate_pct'] = (winning_trades / total_trades) * 100
                metrics['total_trades'] = total_trades

                # PBI-031: Average Trade P&L
                if len(trade_pnls) > 0:
                    metrics['avg_trade_pnl'] = np.mean(trade_pnls)
                    metrics['avg_trade_pnl_pct'] = np.mean(trade_pnls) * 100
            else:
                metrics['win_rate'] = 0.0
                metrics['win_rate_pct'] = 0.0
                metrics['total_trades'] = 0
                metrics['avg_trade_pnl'] = 0.0
                metrics['avg_trade_pnl_pct'] = 0.0
        else:
            metrics['win_rate'] = 0.0
            metrics['win_rate_pct'] = 0.0
            metrics['total_trades'] = 0
            metrics['avg_trade_pnl'] = 0.0
            metrics['avg_trade_pnl_pct'] = 0.0
    else:
        metrics['win_rate'] = 0.0
        metrics['win_rate_pct'] = 0.0
        metrics['total_trades'] = 0
        metrics['avg_trade_pnl'] = 0.0
        metrics['avg_trade_pnl_pct'] = 0.0

    # PBI-031: Advanced Metrics

    # Sortino Ratio (only penalize downside volatility)
    if len(returns) > 0:
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (np.mean(returns) * 8760 - risk_free_rate) / (downside_std * np.sqrt(8760))
            metrics['sortino_ratio'] = sortino_ratio
        else:
            metrics['sortino_ratio'] = 0.0
    else:
        metrics['sortino_ratio'] = 0.0

    # Calmar Ratio (Return / Max Drawdown)
    if max_drawdown < 0:
        calmar_ratio = total_return / abs(max_drawdown)
        metrics['calmar_ratio'] = calmar_ratio
    else:
        metrics['calmar_ratio'] = 0.0

    # Turnover (total volume traded / average portfolio value)
    if trade_blotter is not None and len(trade_blotter) > 0:
        total_volume = trade_blotter['portfolio_value'].sum()
        avg_portfolio_value = np.mean(portfolio_values)
        if avg_portfolio_value > 0:
            turnover = total_volume / avg_portfolio_value
            metrics['turnover'] = turnover
        else:
            metrics['turnover'] = 0.0
    else:
        metrics['turnover'] = 0.0

    return metrics


def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comparison table for all strategies.

    PBI-032: RL Agent vs Benchmarks comparison.

    Args:
        results: Dictionary with results from each strategy

    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Creating comparison table...")

    comparison_data = []

    for strategy_name, result in results.items():
        metrics = result['metrics']

        row = {
            'Strategy': strategy_name,
            'Total Return (%)': f"{metrics.get('total_return_pct', 0):.2f}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.2f}",
            'Max Drawdown (%)': f"{metrics.get('max_drawdown_pct', 0):.2f}",
            'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.2f}",
            'Win Rate (%)': f"{metrics.get('win_rate_pct', 0):.2f}",
            'Total Trades': int(metrics.get('total_trades', 0)),
            'Avg Trade P&L (%)': f"{metrics.get('avg_trade_pnl_pct', 0):.2f}",
            'Turnover': f"{metrics.get('turnover', 0):.2f}"
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df


def plot_price_and_trades(prices: np.ndarray, actions: np.ndarray,
                         title: str = "Price and Trade Points",
                         save_path: Optional[str] = None):
    """
    Plot price chart with buy/sell markers.

    PBI-033: Fiyat ve trade noktaları grafiği.

    Args:
        prices: Array of prices
        actions: Array of actions (0: hold, 1: buy, 2: sell)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot price
    ax.plot(prices, label='BTC Price', linewidth=1.5, color='black', alpha=0.7)

    # Mark buy points
    buy_points = np.where(actions == 1)[0]
    if len(buy_points) > 0:
        ax.scatter(buy_points, prices[buy_points], color='green', marker='^',
                  s=100, label='Buy', zorder=5)

    # Mark sell points
    sell_points = np.where(actions == 2)[0]
    if len(sell_points) > 0:
        ax.scatter(sell_points, prices[sell_points], color='red', marker='v',
                  s=100, label='Sell', zorder=5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price (USD)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_portfolio_values(results: Dict[str, Dict[str, Any]],
                         initial_cash: float = 10000.0,
                         save_path: Optional[str] = None):
    """
    Plot portfolio value curves for all strategies.

    PBI-034: Portfolio değeri eğrisi ile benchmark karşılaştırması.

    Args:
        results: Dictionary with results from each strategy
        initial_cash: Initial capital
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, (strategy_name, result) in enumerate(results.items()):
        portfolio_values = result['portfolio_values']
        returns = (portfolio_values - initial_cash) / initial_cash * 100

        ax.plot(returns, label=strategy_name, linewidth=2,
               color=colors[idx % len(colors)], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Return (%)')
    ax.set_title('Portfolio Value Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_drawdown(portfolio_values: np.ndarray, title: str = "Drawdown",
                 save_path: Optional[str] = None):
    """
    Plot underwater (drawdown) chart.

    PBI-035: Drawdown grafiği (underwater plot).

    Args:
        portfolio_values: Array of portfolio values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100

    # Plot
    ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown, color='red', linewidth=1.5)

    # Mark max drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd_value = drawdown[max_dd_idx]
    ax.scatter([max_dd_idx], [max_dd_value], color='darkred', s=100, zorder=5,
              label=f'Max DD: {max_dd_value:.2f}%')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def export_trade_blotter(trade_blotter: pd.DataFrame, filename: str):
    """
    Export trade blotter to CSV.

    PBI-036: Trade blotter CSV export.

    Args:
        trade_blotter: DataFrame with trade history
        filename: Output CSV filename
    """
    output_path = Path("artifacts") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trade_blotter.to_csv(output_path, index=False)
    logger.info(f"Trade blotter exported to {output_path}")


def evaluate(model_path: str,
            env_config_path: str = "configs/env.yaml",
            training_config_path: str = "configs/training.yaml",
            features_config_path: str = "configs/features.yaml",
            output_dir: str = "artifacts/evaluation"):
    """
    Main evaluation function - Complete Faz 4 pipeline.

    Runs RL agent and benchmarks, calculates metrics, generates visualizations.

    Args:
        model_path: Path to trained model checkpoint
        env_config_path: Path to environment config
        training_config_path: Path to training config
        features_config_path: Path to features config
        output_dir: Directory for output files
    """
    logger.info("="*70)
    logger.info("FAZ 4: EVALUATION AND BACKTEST")
    logger.info("="*70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and validate configs
    logger.info("\n[1/8] Loading configurations...")
    configs = validate_all_configs(env_config_path, training_config_path, features_config_path)
    env_config = configs['env']

    # Prepare test data
    logger.info("\n[2/8] Preparing test data...")
    data_file = env_config['data']['data_file']
    train_df, val_df, test_df = prepare_data_pipeline(
        file_path=data_file,
        add_indicators=True,
        normalize=False,
        train_ratio=0.6,
        val_ratio=0.2
    )
    logger.info(f"Test data: {len(test_df)} samples")

    # Load trained model
    logger.info("\n[3/8] Loading trained model...")
    model = load_model(model_path)

    # Create test environment
    logger.info("\n[4/8] Creating test environment...")
    test_env = DummyVecEnv([lambda: BtcUsdTradingEnv(
        df=test_df,
        window_size=env_config['environment']['window_size'],
        initial_cash=env_config['environment']['initial_cash'],
        transaction_cost=env_config['environment']['transaction_cost'],
        slippage=env_config['environment'].get('slippage', 0.0005),
        max_steps=None  # Use all test data
    )])

    # Dictionary to store all results
    results = {}
    initial_cash = env_config['environment']['initial_cash']

    # Run RL Agent
    logger.info("\n[5/8] Running RL Agent...")
    rl_result = run_episode_rl_agent(test_env, model, deterministic=True)
    rl_metrics = calculate_metrics(
        rl_result['portfolio_values'],
        initial_cash,
        rl_result['trade_blotter']
    )
    results['RL Agent'] = {**rl_result, 'metrics': rl_metrics}

    # Run Buy & Hold Benchmark
    logger.info("\n[6/8] Running Buy & Hold benchmark...")
    bh_result = buy_and_hold_benchmark(test_df, initial_cash,
                                       env_config['environment']['transaction_cost'])
    bh_metrics = calculate_metrics(
        bh_result['portfolio_values'],
        initial_cash,
        bh_result['trade_blotter']
    )
    results['Buy & Hold'] = {**bh_result, 'metrics': bh_metrics}

    # Run RSI Baseline
    logger.info("\n[7/8] Running RSI baseline strategy...")
    rsi_result = rsi_baseline_strategy(test_df, initial_cash,
                                       env_config['environment']['transaction_cost'])
    rsi_metrics = calculate_metrics(
        rsi_result['portfolio_values'],
        initial_cash,
        rsi_result['trade_blotter']
    )
    results['RSI Strategy'] = {**rsi_result, 'metrics': rsi_metrics}

    # Generate reports and visualizations
    logger.info("\n[8/8] Generating reports and visualizations...")

    # Comparison table
    comparison_df = create_comparison_table(results)

    # Print comparison table
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*70)
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison table
    comparison_path = output_path / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison table saved to {comparison_path}")

    # Export trade blotters
    for strategy_name, result in results.items():
        safe_name = strategy_name.lower().replace(' ', '_').replace('&', 'and')
        export_trade_blotter(
            result['trade_blotter'],
            f"trade_blotter_{safe_name}.csv"
        )

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    # Price and trades for RL Agent
    plot_price_and_trades(
        results['RL Agent']['prices'],
        results['RL Agent']['actions'],
        title="RL Agent: Price and Trade Points",
        save_path=output_path / "rl_agent_trades.png"
    )

    # Portfolio values comparison
    plot_portfolio_values(
        results,
        initial_cash,
        save_path=output_path / "portfolio_comparison.png"
    )

    # Drawdown for RL Agent
    plot_drawdown(
        results['RL Agent']['portfolio_values'],
        title="RL Agent Drawdown",
        save_path=output_path / "rl_agent_drawdown.png"
    )

    # Drawdown for Buy & Hold
    plot_drawdown(
        results['Buy & Hold']['portfolio_values'],
        title="Buy & Hold Drawdown",
        save_path=output_path / "buy_hold_drawdown.png"
    )

    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nAll outputs saved to: {output_path}")
    logger.info(f"  - Comparison table: comparison_table.csv")
    logger.info(f"  - Trade blotters: trade_blotter_*.csv")
    logger.info(f"  - Visualizations: *.png")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    for strategy_name, result in results.items():
        metrics = result['metrics']
        logger.info(f"\n{strategy_name}:")
        logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        logger.info(f"  Total Trades: {metrics['total_trades']}")

    test_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent and benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.zip file)"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env.yaml",
        help="Path to environment config file"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--features-config",
        type=str,
        default="configs/features.yaml",
        help="Path to features config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory for evaluation outputs"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        env_config_path=args.env_config,
        training_config_path=args.training_config,
        features_config_path=args.features_config,
        output_dir=args.output_dir
    )
