"""
pytest test suite for trading_env.py

Epic 2.3: PBI-016 and PBI-017
Comprehensive tests for BtcUsdTradingEnv

Run with: pytest test_trading_env_pytest.py -v
"""

import pandas as pd
import numpy as np
import pytest
from gymnasium import spaces

from trading_env import BtcUsdTradingEnv
from data_manager import add_technical_indicators


# Fixtures
@pytest.fixture
def sample_market_data():
    """Create sample market data with technical indicators."""
    n_samples = 200
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H', tz='UTC')

    np.random.seed(42)
    base_price = 30000
    close = base_price + np.cumsum(np.random.randn(n_samples) * 100)
    open_prices = close * (1 + np.random.randn(n_samples) * 0.001)
    high = np.maximum(open_prices, close) * 1.01
    low = np.minimum(open_prices, close) * 0.99
    volume = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Add technical indicators
    df = add_technical_indicators(df)

    return df


@pytest.fixture
def env(sample_market_data):
    """Create a basic trading environment."""
    return BtcUsdTradingEnv(
        df=sample_market_data,
        window_size=24,
        initial_cash=10000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        max_steps=50
    )


# Test Environment Initialization (PBI-016)
class TestEnvironmentInitialization:
    """Test suite for environment initialization."""

    def test_init_valid_params(self, sample_market_data):
        """Test environment initialization with valid parameters."""
        env = BtcUsdTradingEnv(
            df=sample_market_data,
            window_size=24,
            initial_cash=10000.0
        )

        assert env.window_size == 24
        assert env.initial_cash == 10000.0
        assert env.transaction_cost == 0.001
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_init_dataframe_too_short(self):
        """Test that short dataframes raise ValueError."""
        df = pd.DataFrame({
            'open': [1, 2],
            'high': [3, 4],
            'low': [0.5, 1],
            'close': [2, 3],
            'volume': [100, 200]
        })

        with pytest.raises(ValueError, match="at least"):
            BtcUsdTradingEnv(df=df, window_size=10)

    def test_init_missing_columns(self):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            # Missing 'low', 'close', 'volume'
        })

        with pytest.raises(ValueError, match="missing required columns"):
            BtcUsdTradingEnv(df=df, window_size=2)

    def test_action_space(self, env):
        """Test that action space is correctly defined."""
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3  # Hold, Buy, Sell

    def test_observation_space(self, env):
        """Test that observation space is correctly defined."""
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (env.window_size, env.n_features)
        assert env.observation_space.dtype == np.float32


# Test Reset Functionality (PBI-016)
class TestReset:
    """Test suite for reset() method."""

    def test_reset_returns_correct_types(self, env):
        """Test that reset returns observation and info dict."""
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape(self, env):
        """Test that reset returns correctly shaped observation."""
        obs, info = env.reset()

        assert obs.shape == (env.window_size, env.n_features)
        assert obs.dtype == np.float32

    def test_reset_initializes_state(self, env):
        """Test that reset properly initializes state variables."""
        obs, info = env.reset()

        assert env.current_step == 0
        assert env.cash == env.initial_cash
        assert env.position == 0
        assert env.position_price == 0.0
        assert env.total_reward == 0.0
        assert len(env.portfolio_values) == 1

    def test_reset_info_dict(self, env):
        """Test that reset info dict contains expected keys."""
        obs, info = env.reset()

        assert 'step' in info
        assert 'cash' in info
        assert 'position' in info
        assert 'portfolio_value' in info
        assert info['portfolio_value'] == env.initial_cash

    def test_reset_with_seed(self, env):
        """Test that seed produces reproducible resets."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)


# Test Step Functionality (PBI-016)
class TestStep:
    """Test suite for step() method."""

    def test_step_returns_correct_types(self, env):
        """Test that step returns correct tuple types."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, env):
        """Test that step returns correctly shaped observation."""
        env.reset()
        obs, _, _, _, _ = env.step(0)

        assert obs.shape == (env.window_size, env.n_features)
        assert obs.dtype == np.float32

    def test_step_hold_action(self, env):
        """Test that hold action doesn't change position."""
        env.reset()
        initial_cash = env.cash
        initial_position = env.position

        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        assert env.cash == initial_cash
        assert env.position == initial_position
        assert info['transaction_occurred'] is False

    def test_step_buy_action(self, env):
        """Test that buy action opens position."""
        env.reset()
        initial_cash = env.cash

        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        assert env.position == 1  # Long position
        assert env.cash < initial_cash  # Cash reduced by transaction cost
        assert env.position_price > 0
        assert info['transaction_occurred'] is True

    def test_step_sell_action(self, env):
        """Test that sell action closes position."""
        env.reset()
        env.step(1)  # Buy first

        position_before = env.position
        obs, reward, terminated, truncated, info = env.step(2)  # Sell

        assert position_before == 1
        assert env.position == 0  # Position closed
        assert env.position_price == 0.0
        assert info['transaction_occurred'] is True

    def test_step_sell_without_position(self, env):
        """Test that sell without position does nothing."""
        env.reset()

        obs, reward, terminated, truncated, info = env.step(2)  # Sell without position

        assert env.position == 0
        assert info['transaction_occurred'] is False

    def test_step_buy_when_already_long(self, env):
        """Test that buy when already long does nothing."""
        env.reset()
        env.step(1)  # Buy

        position_price = env.position_price
        obs, reward, terminated, truncated, info = env.step(1)  # Try to buy again

        assert env.position == 1
        assert env.position_price == position_price  # Unchanged
        assert info['transaction_occurred'] is False

    def test_step_increments_step_counter(self, env):
        """Test that step increments current_step."""
        env.reset()
        initial_step = env.current_step

        env.step(0)

        assert env.current_step == initial_step + 1

    def test_step_info_dict(self, env):
        """Test that step info dict contains expected keys."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        required_keys = ['step', 'action', 'transaction_occurred', 'current_price',
                        'cash', 'position', 'portfolio_value', 'total_reward']

        for key in required_keys:
            assert key in info, f"Missing key: {key}"


# Test Transaction Costs and Slippage (PBI-016)
class TestCosts:
    """Test suite for transaction costs and slippage."""

    def test_transaction_cost_applied_on_buy(self, env):
        """Test that transaction cost is applied on buy."""
        env.reset()
        initial_cash = env.cash

        env.step(1)  # Buy

        # Cash should be reduced by transaction cost
        expected_cash = initial_cash * (1 - env.transaction_cost)
        assert abs(env.cash - expected_cash) < 0.01  # Small tolerance

    def test_transaction_cost_applied_on_sell(self, env):
        """Test that transaction cost is applied on sell."""
        env.reset()
        env.step(1)  # Buy

        cash_before_sell = env.cash
        current_price = env.df['close'].iloc[env.current_step + env.window_size]

        env.step(2)  # Sell

        # Should get back less than theoretical value due to costs
        # Exact calculation depends on price movement, but cash should have changed
        assert env.cash != cash_before_sell

    def test_slippage_increases_buy_price(self, env):
        """Test that slippage increases execution price on buy."""
        env.reset()
        current_price = env.df['close'].iloc[env.current_step + env.window_size]

        env.step(1)  # Buy

        # Position price should be higher than current price due to slippage
        assert env.position_price > current_price


# Test Episode Termination (PBI-016)
class TestTermination:
    """Test suite for episode termination conditions."""

    def test_truncation_at_max_steps(self, env):
        """Test that episode truncates at max_steps."""
        env.reset()

        terminated = False
        truncated = False

        for _ in range(env.max_steps + 1):
            if terminated or truncated:
                break
            _, _, terminated, truncated, _ = env.step(0)

        assert truncated is True
        assert env.current_step >= env.max_steps

    def test_no_premature_termination(self, env):
        """Test that episode doesn't terminate prematurely."""
        env.reset()

        for _ in range(min(10, env.max_steps)):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        # Should not terminate in first 10 steps with hold actions
        assert env.current_step < env.max_steps


# Test Portfolio Calculations (PBI-016)
class TestPortfolio:
    """Test suite for portfolio value calculations."""

    def test_initial_portfolio_value(self, env):
        """Test that initial portfolio equals initial cash."""
        env.reset()
        current_price = env.df['close'].iloc[env.current_step + env.window_size]
        portfolio_value = env._calculate_portfolio_value(current_price)

        assert portfolio_value == env.initial_cash

    def test_portfolio_value_with_position(self, env):
        """Test portfolio value calculation with open position."""
        env.reset()
        env.step(1)  # Buy

        current_price = env.df['close'].iloc[env.current_step + env.window_size]
        portfolio_value = env._calculate_portfolio_value(current_price)

        # Portfolio should reflect current position value
        assert portfolio_value > 0
        assert portfolio_value != env.cash  # Should include position value

    def test_portfolio_value_tracking(self, env):
        """Test that portfolio values are tracked over time."""
        env.reset()

        for _ in range(5):
            env.step(0)

        assert len(env.portfolio_values) == 6  # Initial + 5 steps


# Manual Trading Simulation (PBI-017)
class TestManualSimulation:
    """Test suite for manual trading simulation."""

    def test_simple_buy_hold_sell_sequence(self, env):
        """Test a simple buy-hold-sell trading sequence."""
        env.reset()

        # Buy
        obs, reward, terminated, truncated, info = env.step(1)
        assert env.position == 1
        buy_price = env.position_price

        # Hold for a few steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)

        # Sell
        obs, reward, terminated, truncated, info = env.step(2)
        assert env.position == 0

        # Portfolio value should have changed
        final_portfolio = info['portfolio_value']
        assert final_portfolio != env.initial_cash

    def test_random_action_sequence(self, env):
        """Test random action sequence doesn't crash."""
        env.reset()
        np.random.seed(42)

        for _ in range(20):
            action = np.random.choice([0, 1, 2])
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Should complete without errors
        assert True

    def test_episode_consistency(self, env):
        """Test that full episode maintains consistency."""
        env.reset()

        total_steps = 0
        portfolio_values = []

        while True:
            action = 0  # Hold for simplicity
            obs, reward, terminated, truncated, info = env.step(action)

            total_steps += 1
            portfolio_values.append(info['portfolio_value'])

            if terminated or truncated:
                break

        # Verify consistency
        assert total_steps == env.current_step
        assert len(portfolio_values) == total_steps
        assert all(pv > 0 for pv in portfolio_values)  # All values positive


# Test Render (PBI-017)
class TestRender:
    """Test suite for render() method."""

    def test_render_after_reset(self, env, capsys):
        """Test that render works after reset."""
        env.reset()
        env.render()

        captured = capsys.readouterr()
        assert "Step:" in captured.out
        assert "Portfolio Value:" in captured.out

    def test_render_before_reset(self, env, capsys):
        """Test that render warns if called before reset."""
        env.render()

        captured = capsys.readouterr()
        # Should log warning (not crash)


# Integration Tests
class TestIntegration:
    """Integration tests for full environment workflow."""

    def test_full_episode_with_trades(self, sample_market_data):
        """Test a complete episode with multiple trades."""
        env = BtcUsdTradingEnv(
            df=sample_market_data,
            window_size=10,
            initial_cash=10000.0,
            max_steps=30
        )

        obs, info = env.reset(seed=42)

        actions = [0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2]  # Predefined sequence

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Check final state
        assert env.current_step > 0
        assert len(env.portfolio_values) == env.current_step + 1

    def test_environment_with_minimal_data(self):
        """Test environment with minimal viable data."""
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [110] * 50,
            'low': [90] * 50,
            'close': [105] * 50,
            'volume': [1000] * 50
        })

        env = BtcUsdTradingEnv(df=df, window_size=5, max_steps=10)
        obs, info = env.reset()

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)

        assert env.current_step == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
