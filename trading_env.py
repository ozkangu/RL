"""
Trading Environment Module
Gymnasium-compatible environment for BTC/USD trading with RL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class BtcUsdTradingEnv(gym.Env):
    """
    Custom Trading Environment for BTC/USD that follows gymnasium interface.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 24,
                 initial_cash: float = 10000.0, transaction_cost: float = 0.001):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with market data
            window_size: Number of historical bars to include in observation
            initial_cash: Initial cash in USD
            transaction_cost: Transaction cost as percentage
        """
        super().__init__()

        # TODO: Implement in Epic 2.1
        pass

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            observation, info dict
        """
        # TODO: Implement in Epic 2.2
        pass

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        # TODO: Implement in Epic 2.2
        pass

    def render(self):
        """Render the environment."""
        # TODO: Implement later if needed
        pass
