"""
Configuration Validator Module
Validates YAML configuration files to ensure all required fields are present and valid.
"""

import yaml
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration

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

    return config


def validate_env_config(config: Dict[str, Any]) -> None:
    """
    Validate environment configuration (env.yaml).

    Args:
        config: Configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    required_sections = ['data', 'environment', 'action', 'reward', 'episode', 'normalization']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")

    # Validate data section
    if 'data_file' not in config['data']:
        raise ConfigValidationError("Missing 'data_file' in data section")

    # Validate environment section
    env = config['environment']
    required_env_fields = ['window_size', 'initial_cash', 'transaction_cost']
    for field in required_env_fields:
        if field not in env:
            raise ConfigValidationError(f"Missing required field in environment: {field}")

    # Validate numeric values
    if env['window_size'] <= 0:
        raise ConfigValidationError("window_size must be positive")
    if env['initial_cash'] <= 0:
        raise ConfigValidationError("initial_cash must be positive")
    if not (0 <= env['transaction_cost'] < 1):
        raise ConfigValidationError("transaction_cost must be in [0, 1)")

    # Validate action section
    if 'type' not in config['action']:
        raise ConfigValidationError("Missing 'type' in action section")

    action_type = config['action']['type']
    if action_type not in ['discrete', 'continuous']:
        raise ConfigValidationError(f"Invalid action type: {action_type}")

    logger.info("Environment config validation passed")


def validate_training_config(config: Dict[str, Any]) -> None:
    """
    Validate training configuration (training.yaml).

    Args:
        config: Configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    required_sections = ['algorithm', 'ppo', 'training', 'checkpoint', 'logging']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")

    # Validate algorithm
    if config['algorithm']['name'] not in ['PPO', 'SAC', 'A2C']:
        raise ConfigValidationError(f"Unsupported algorithm: {config['algorithm']['name']}")

    # Validate PPO hyperparameters
    ppo = config['ppo']
    required_ppo_fields = ['learning_rate', 'n_steps', 'batch_size', 'gamma']
    for field in required_ppo_fields:
        if field not in ppo:
            raise ConfigValidationError(f"Missing required PPO field: {field}")

    # Validate numeric ranges
    if not (0 < ppo['learning_rate'] < 1):
        raise ConfigValidationError("learning_rate must be in (0, 1)")
    if ppo['n_steps'] <= 0:
        raise ConfigValidationError("n_steps must be positive")
    if ppo['batch_size'] <= 0:
        raise ConfigValidationError("batch_size must be positive")
    if not (0 < ppo['gamma'] <= 1):
        raise ConfigValidationError("gamma must be in (0, 1]")

    # Validate training settings
    training = config['training']
    if training['total_timesteps'] <= 0:
        raise ConfigValidationError("total_timesteps must be positive")

    # Check batch_size <= n_steps
    if ppo['batch_size'] > ppo['n_steps']:
        raise ConfigValidationError("batch_size cannot be larger than n_steps")

    logger.info("Training config validation passed")


def validate_features_config(config: Dict[str, Any]) -> None:
    """
    Validate features configuration (features.yaml).

    Args:
        config: Configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    required_sections = ['base_features', 'indicators', 'data_quality']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")

    # Validate base features
    required_base = ['open', 'high', 'low', 'close', 'volume']
    if not all(feat in config['base_features'] for feat in required_base):
        raise ConfigValidationError(f"base_features must include OHLCV: {required_base}")

    # Validate indicators
    indicators = config['indicators']
    for indicator_name, indicator_config in indicators.items():
        if not isinstance(indicator_config, dict):
            raise ConfigValidationError(f"Indicator config must be a dict: {indicator_name}")
        if 'enabled' not in indicator_config:
            raise ConfigValidationError(f"Missing 'enabled' field for indicator: {indicator_name}")

    logger.info("Features config validation passed")


def validate_all_configs(env_path: str = "configs/env.yaml",
                         training_path: str = "configs/training.yaml",
                         features_path: str = "configs/features.yaml") -> Dict[str, Dict[str, Any]]:
    """
    Validate all configuration files.

    Args:
        env_path: Path to environment config
        training_path: Path to training config
        features_path: Path to features config

    Returns:
        Dictionary with all loaded and validated configs

    Raises:
        ConfigValidationError: If any validation fails
    """
    logger.info("Starting configuration validation...")

    configs = {}

    try:
        # Load and validate environment config
        env_config = load_yaml_config(env_path)
        validate_env_config(env_config)
        configs['env'] = env_config

        # Load and validate training config
        training_config = load_yaml_config(training_path)
        validate_training_config(training_config)
        configs['training'] = training_config

        # Load and validate features config
        features_config = load_yaml_config(features_path)
        validate_features_config(features_config)
        configs['features'] = features_config

        logger.info("All configuration files validated successfully")

    except (FileNotFoundError, yaml.YAMLError, ConfigValidationError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    return configs


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    try:
        configs = validate_all_configs()
        print("✓ All configurations are valid!")
        print(f"\nLoaded configs: {list(configs.keys())}")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
