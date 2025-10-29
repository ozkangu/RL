"""
Training Script
Trains RL agent on BTC/USD trading environment.

Epic 3.1 & 3.2: Complete training pipeline with checkpointing and monitoring.
"""

import argparse
import yaml
from pathlib import Path
import logging
import numpy as np
import torch
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

from data_manager import prepare_data_pipeline
from trading_env import BtcUsdTradingEnv
from config_validator import validate_all_configs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    PBI-018: Config loading from YAML files.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    logger.info(f"Loaded configuration from {config_path}")
    return config


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    PBI-024: Reproducibility settings.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    IMPORTANT FIX: Adaptive learning rate schedule for better convergence.

    Args:
        initial_value: Initial learning rate

    Returns:
        Schedule function that takes progress_remaining (1.0 to 0.0) and returns LR
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        LR will decrease proportionally.
        """
        return progress_remaining * initial_value

    return func


def make_env(df, env_config: Dict[str, Any], rank: int = 0, seed: int = 0):
    """
    Create a single trading environment (for vectorization).

    Args:
        df: Market data DataFrame
        env_config: Environment configuration
        rank: Environment rank (for parallel envs)
        seed: Random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        env = BtcUsdTradingEnv(
            df=df,
            window_size=env_config['environment']['window_size'],
            initial_cash=env_config['environment']['initial_cash'],
            transaction_cost=env_config['environment']['transaction_cost'],
            slippage=env_config['environment'].get('slippage', 0.0005),
            max_steps=env_config['episode'].get('max_steps', None)
        )
        env.reset(seed=seed + rank)
        # Wrap with Monitor for logging
        env = Monitor(env)
        return env
    return _init


def train(training_config_path: str = "configs/training.yaml",
          env_config_path: str = "configs/env.yaml",
          features_config_path: str = "configs/features.yaml"):
    """
    Main training function.

    Epic 3.1 & 3.2: Complete training pipeline with:
    - Config loading and validation
    - Data preparation
    - Environment setup
    - PPO agent initialization
    - Training loop with callbacks
    - Checkpointing and monitoring

    Args:
        training_config_path: Path to training config
        env_config_path: Path to environment config
        features_config_path: Path to features config
    """
    logger.info("="*70)
    logger.info("BTC/USD RL TRADING AGENT TRAINING")
    logger.info("="*70)

    # PBI-018: Load and validate all configurations
    logger.info("\n[1/7] Loading and validating configurations...")
    try:
        configs = validate_all_configs(
            env_path=env_config_path,
            training_path=training_config_path,
            features_path=features_config_path
        )
        training_config = configs['training']
        env_config = configs['env']
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # PBI-024: Set random seed for reproducibility
    seed = training_config['training']['seed']
    set_seed(seed)

    # PBI-018: Prepare data
    logger.info("\n[2/7] Preparing data...")
    data_file = env_config['data']['data_file']
    train_df, val_df, test_df = prepare_data_pipeline(
        file_path=data_file,
        add_indicators=True,
        normalize=False,
        train_ratio=0.6,
        val_ratio=0.2
    )

    logger.info(f"Training data: {len(train_df)} samples")
    logger.info(f"Validation data: {len(val_df)} samples")
    logger.info(f"Test data: {len(test_df)} samples")

    # PBI-020: Create vectorized environments (DummyVecEnv for MVP)
    logger.info("\n[3/7] Creating training environment...")
    n_envs = training_config['training']['n_envs']

    if n_envs == 1:
        # Single environment wrapped in DummyVecEnv
        env = DummyVecEnv([make_env(train_df, env_config, 0, seed)])
        logger.info("Using DummyVecEnv with 1 environment")
    else:
        # Multiple environments
        env = DummyVecEnv([make_env(train_df, env_config, i, seed) for i in range(n_envs)])
        logger.info(f"Using DummyVecEnv with {n_envs} environments")

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(val_df, env_config, 0, seed + 1000)])

    # PBI-019: Initialize PPO agent
    logger.info("\n[4/7] Initializing PPO agent...")
    ppo_config = training_config['ppo']

    # IMPORTANT FIX: Linear learning rate schedule for better convergence
    use_lr_schedule = training_config['training'].get('use_lr_schedule', True)
    if use_lr_schedule:
        learning_rate = linear_schedule(ppo_config['learning_rate'])
        logger.info(f"Using linear LR schedule (initial: {ppo_config['learning_rate']})")
    else:
        learning_rate = ppo_config['learning_rate']
        logger.info(f"Using fixed LR: {ppo_config['learning_rate']}")

    # NICE-TO-HAVE FIX: Policy architecture from config
    policy_kwargs = training_config.get('policy_kwargs', {
        'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
    })

    model = PPO(
        policy=training_config['algorithm']['policy'],
        env=env,
        learning_rate=learning_rate,
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=training_config['logging']['verbose'],
        tensorboard_log=training_config['logging']['tensorboard_log'],
        seed=seed
    )

    logger.info("PPO agent initialized with hyperparameters:")
    logger.info(f"  Learning rate: {ppo_config['learning_rate']} (schedule: {use_lr_schedule})")
    logger.info(f"  n_steps: {ppo_config['n_steps']}")
    logger.info(f"  batch_size: {ppo_config['batch_size']}")
    logger.info(f"  gamma: {ppo_config['gamma']}")
    logger.info(f"  ent_coef: {ppo_config['ent_coef']}")
    logger.info(f"  policy_kwargs: {policy_kwargs}")

    # PBI-022: Setup callbacks for checkpointing
    logger.info("\n[5/7] Setting up callbacks...")

    # Create checkpoint directory
    checkpoint_dir = Path(training_config['checkpoint']['save_path'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['checkpoint']['save_freq'],
        save_path=str(checkpoint_dir),
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # IMPORTANT FIX: Early stopping callback
    # Stops training if no improvement in eval reward for N evaluations
    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=training_config['training'].get('early_stop_patience', 5),
        min_evals=training_config['training'].get('early_stop_min_evals', 10),
        verbose=1
    )

    # Evaluation callback - evaluates and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=training_config['checkpoint']['save_freq'],
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        callback_after_eval=early_stop_callback  # NEW: Early stopping
    )

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback])

    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"Save frequency: every {training_config['checkpoint']['save_freq']} steps")
    logger.info(f"Early stopping: patience={training_config['training'].get('early_stop_patience', 5)} evals")

    # PBI-021: Training loop
    logger.info("\n[6/7] Starting training...")
    logger.info("="*70)

    total_timesteps = training_config['training']['total_timesteps']
    logger.info(f"Training for {total_timesteps:,} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=training_config['logging']['log_interval'],
            progress_bar=True
        )

        logger.info("\n" + "="*70)
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        raise

    # Save final model
    logger.info("\n[7/7] Saving final model...")
    final_model_path = checkpoint_dir / "final_model"
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}.zip")

    # Cleanup
    env.close()
    eval_env.close()

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*70)
    logger.info(f"\nArtifacts saved:")
    logger.info(f"  - Checkpoints: {checkpoint_dir}")
    logger.info(f"  - Best model: {checkpoint_dir / 'best_model'}")
    logger.info(f"  - Final model: {final_model_path}.zip")
    logger.info(f"  - TensorBoard logs: {training_config['logging']['tensorboard_log']}")
    logger.info(f"\nView training logs with:")
    logger.info(f"  tensorboard --logdir {training_config['logging']['tensorboard_log']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agent for BTC/USD trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env.yaml",
        help="Path to environment config file"
    )
    parser.add_argument(
        "--features-config",
        type=str,
        default="configs/features.yaml",
        help="Path to features config file"
    )

    args = parser.parse_args()

    train(
        training_config_path=args.training_config,
        env_config_path=args.env_config,
        features_config_path=args.features_config
    )
