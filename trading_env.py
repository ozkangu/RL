"""
Trading Environment Module
Gymnasium-compatible environment for BTC/USD trading with RL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BtcUsdTradingEnv(gym.Env):
    """
    Custom Trading Environment for BTC/USD that follows gymnasium interface.

    PBI-009, PBI-010, PBI-011: Gymnasium-compatible trading environment with
    discrete action space and market observation space.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 24,
                 initial_cash: float = 10000.0, transaction_cost: float = 0.001,
                 slippage: float = 0.0005, max_steps: Optional[int] = None):
        """
        Initialize trading environment.

        PBI-009: Environment initialization with all required parameters.

        Args:
            df: DataFrame with market data (must include OHLCV and indicators)
            window_size: Number of historical bars to include in observation
            initial_cash: Initial cash in USD
            transaction_cost: Transaction cost as percentage (e.g., 0.001 = 0.1%)
            slippage: Slippage as percentage (e.g., 0.0005 = 0.05%)
            max_steps: Maximum steps per episode (None = use all data)

        Raises:
            ValueError: If df is too short or missing required columns
        """
        super().__init__()

        # Validate input
        if len(df) < window_size + 1:
            raise ValueError(f"DataFrame must have at least {window_size + 1} rows")

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        # Store configuration
        self.df = df.reset_index(drop=False)  # Keep index as column if datetime
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_steps = max_steps if max_steps is not None else len(df) - window_size

        # Feature columns (all numeric columns except timestamp)
        self.feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.n_features = len(self.feature_columns)

        logger.info(f"Trading environment initialized:")
        logger.info(f"  Data shape: {self.df.shape}")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Initial cash: ${self.initial_cash:,.2f}")
        logger.info(f"  Transaction cost: {self.transaction_cost:.4f}")
        logger.info(f"  Max steps per episode: {self.max_steps}")

        # PBI-010: Define action space
        # 0: Hold (do nothing)
        # 1: Buy (or hold if already long)
        # 2: Sell (close position if long)
        self.action_space = spaces.Discrete(3)

        # PBI-011: Define observation space
        # Observation is a window of historical market data
        # Shape: (window_size, n_features)
        # Each row contains: OHLCV + technical indicators
        # Note: Using reasonable bounds instead of -inf/inf for better NN training
        self.observation_space = spaces.Box(
            low=-10.0,  # Reasonable bound for normalized features
            high=10.0,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )

        # Initialize state variables (will be set in reset())
        self.current_step = None
        self.invested_capital = None  # Capital invested in position (renamed from 'cash' for clarity)
        self.position = None  # 0: no position, 1: long position
        self.position_price = None  # Entry price when position was opened
        self.total_reward = None
        self.portfolio_values = None

        logger.info(f"Action space: Discrete(3) - Hold/Buy/Sell")
        logger.info(f"Observation space: Box({self.window_size}, {self.n_features})")

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (window of historical market data).

        Returns:
            Numpy array of shape (window_size, n_features) containing market data

        Raises:
            IndexError: If not enough data available for observation window
        """
        # Get the window of data ending at current_step
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size

        # CRITICAL FIX: Check bounds to prevent index out of range errors
        if end_idx > len(self.df):
            raise IndexError(
                f"Insufficient data for observation. "
                f"Requested indices [{start_idx}:{end_idx}], "
                f"but DataFrame only has {len(self.df)} rows."
            )

        # Extract feature values
        obs = self.df[self.feature_columns].iloc[start_idx:end_idx].values

        # Verify shape
        if obs.shape[0] != self.window_size:
            raise ValueError(
                f"Observation shape mismatch. Expected {self.window_size} rows, "
                f"got {obs.shape[0]} rows."
            )

        # Convert to float32 for neural network compatibility
        return obs.astype(np.float32)

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calculate current portfolio value.

        IMPORTANT: When we have a position, self.invested_capital represents
        the capital that was used to buy BTC at self.position_price.
        We calculate current value by seeing what that BTC is worth now.

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value
        """
        if self.position == 0:
            # No position, portfolio value is just our capital
            return self.invested_capital
        else:
            # Has position: Calculate current value of our BTC holdings
            # Our invested_capital was used to buy BTC at position_price
            btc_amount = self.invested_capital / self.position_price
            # Current value of that BTC at market price
            position_value = btc_amount * current_price
            return position_value

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        PBI-012: Reset environment to beginning of episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used currently)

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        # Set seed for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset state variables
        self.current_step = 0
        self.invested_capital = self.initial_cash
        self.position = 0  # Start with no position
        self.position_price = 0.0
        self.total_reward = 0.0
        self.portfolio_values = [self.initial_cash]

        # Get initial observation
        obs = self._get_observation()

        # Info dictionary
        info = {
            'step': self.current_step,
            'cash': self.invested_capital,
            'position': self.position,
            'portfolio_value': self.initial_cash
        }

        logger.debug(f"Environment reset at step {self.current_step}")

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        PBI-013, PBI-014, PBI-015: Execute action, apply costs, calculate reward,
        and check episode termination.

        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated (max steps)
            info: Additional information dictionary
        """
        # Get current price (at the end of current bar, T+0 execution)
        current_price = self.df['close'].iloc[self.current_step + self.window_size]

        # Store portfolio value before action
        portfolio_value_before = self._calculate_portfolio_value(current_price)

        # PBI-013: Execute action with T+0 execution (simplified for MVP)
        # PBI-014: Apply transaction costs and slippage
        transaction_occurred = False

        if action == 1:  # Buy
            if self.position == 0:  # Only buy if we don't have a position
                # Apply slippage (we buy at a slightly higher price)
                execution_price = current_price * (1 + self.slippage)

                # Apply transaction cost
                # Cost is deducted from capital available for position
                cost_factor = 1 - self.transaction_cost
                effective_capital = self.invested_capital * cost_factor

                # Open long position
                self.position = 1
                self.position_price = execution_price
                self.invested_capital = effective_capital

                transaction_occurred = True
                logger.debug(f"BUY at ${execution_price:,.2f} (slippage: {self.slippage:.4f}, cost: {self.transaction_cost:.4f})")

        elif action == 2:  # Sell
            if self.position == 1:  # Only sell if we have a position
                # Apply slippage (we sell at a slightly lower price)
                execution_price = current_price * (1 - self.slippage)

                # Calculate BTC amount we have
                btc_amount = self.invested_capital / self.position_price

                # Sell BTC and get capital back
                capital_from_sale = btc_amount * execution_price

                # Apply transaction cost
                cost_factor = 1 - self.transaction_cost
                self.invested_capital = capital_from_sale * cost_factor

                # Close position
                self.position = 0
                self.position_price = 0.0

                transaction_occurred = True
                logger.debug(f"SELL at ${execution_price:,.2f} (slippage: {self.slippage:.4f}, cost: {self.transaction_cost:.4f})")

        # Action 0 (Hold) does nothing

        # Calculate portfolio value after action
        portfolio_value_after = self._calculate_portfolio_value(current_price)

        # Calculate reward (log return of portfolio value)
        # IMPORTANT FIX: Added clipping to prevent extreme rewards
        if portfolio_value_before > 0:
            raw_reward = np.log(portfolio_value_after / portfolio_value_before)
            # Clip to reasonable range to prevent exploding/vanishing gradients
            reward = np.clip(raw_reward, -0.1, 0.1)
        else:
            reward = 0.0  # Avoid log(0)

        self.total_reward += reward
        self.portfolio_values.append(portfolio_value_after)

        # Move to next step
        self.current_step += 1

        # Get next observation
        obs = self._get_observation()

        # PBI-015: Check episode termination
        terminated = False
        truncated = False

        # Episode terminates if portfolio value drops to zero or negative
        if portfolio_value_after <= 0:
            terminated = True
            logger.info(f"Episode terminated: Portfolio value <= 0 (${portfolio_value_after:,.2f})")

        # Episode is truncated if we reach max steps
        if self.current_step >= self.max_steps:
            truncated = True
            logger.info(f"Episode truncated: Reached max steps ({self.max_steps})")

        # Info dictionary with detailed information
        info = {
            'step': self.current_step,
            'action': action,
            'transaction_occurred': transaction_occurred,
            'current_price': current_price,
            'cash': self.invested_capital,
            'position': self.position,
            'position_price': self.position_price,
            'portfolio_value': portfolio_value_after,
            'total_reward': self.total_reward,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the environment state (for debugging/visualization).

        Prints current step, portfolio value, position, and cash.
        """
        if self.current_step is None:
            logger.warning("Environment not initialized. Call reset() first.")
            return

        current_price = self.df['close'].iloc[self.current_step + self.window_size]
        portfolio_value = self._calculate_portfolio_value(current_price)

        print(f"\n{'='*50}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Invested Capital: ${self.invested_capital:,.2f}")
        print(f"Position: {'Long' if self.position == 1 else 'None'}")
        if self.position == 1:
            print(f"Entry Price: ${self.position_price:,.2f}")
            pnl = ((current_price - self.position_price) / self.position_price) * 100
            print(f"P&L: {pnl:+.2f}%")
        print(f"Total Reward: {self.total_reward:.4f}")
        print(f"{'='*50}\n")
