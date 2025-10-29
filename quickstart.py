#!/usr/bin/env python3
"""
Quick Start Script for RL Trading Bot
Runs the complete pipeline: data preparation, training, and evaluation.

Usage:
    python quickstart.py --mode demo    # Quick demo with sample data (10k steps)
    python quickstart.py --mode full    # Full training with real data (200k steps)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, total_steps, text):
    """Print formatted step."""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed!")
        return False
    print(f"✅ {description} completed successfully")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    print("Checking dependencies...")
    try:
        import gymnasium
        import stable_baselines3
        import pandas
        import numpy
        import ta
        import matplotlib
        import yaml
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

def setup_demo_config():
    """Create demo configuration with reduced timesteps."""
    import yaml

    print("Setting up demo configuration...")

    # Read original config
    with open('configs/training.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Modify for demo
    config['training']['total_timesteps'] = 10000  # Reduced for demo
    config['checkpoint']['save_freq'] = 2000

    # Save demo config
    demo_config_path = 'configs/training_demo.yaml'
    with open(demo_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Demo config created: {demo_config_path}")
    return demo_config_path

def setup_demo_env_config():
    """Create demo environment config with sample data."""
    import yaml

    # Read original config
    with open('configs/env.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use sample data
    config['data']['data_file'] = 'data/sample_btcusdt_1h.csv'

    # Save demo config
    demo_env_path = 'configs/env_demo.yaml'
    with open(demo_env_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Demo env config created: {demo_env_path}")
    return demo_env_path

def quickstart_demo():
    """Run quick demo with sample data."""
    print_header("QUICK START - DEMO MODE")
    print("This will:")
    print("  - Use sample BTC data (1000 hours)")
    print("  - Train for 10,000 timesteps (~2-5 minutes)")
    print("  - Evaluate the model")
    print("  - Generate reports and visualizations")

    input("\nPress Enter to continue (or Ctrl+C to cancel)...")

    total_steps = 6

    # Step 1: Check dependencies
    print_step(1, total_steps, "Checking dependencies")
    if not check_dependencies():
        return False

    # Step 2: Generate sample data if not exists
    print_step(2, total_steps, "Preparing sample data")
    sample_data = Path('data/sample_btcusdt_1h.csv')
    if not sample_data.exists():
        print("Sample data not found, generating...")
        if not run_command('python generate_sample_data.py', 'Sample data generation'):
            return False
    else:
        print(f"✅ Sample data exists: {sample_data}")

    # Step 3: Setup demo configs
    print_step(3, total_steps, "Setting up demo configuration")
    demo_training_config = setup_demo_config()
    demo_env_config = setup_demo_env_config()

    # Step 4: Validate configuration
    print_step(4, total_steps, "Validating configuration")
    if not run_command('python config_validator.py', 'Config validation'):
        return False

    # Step 5: Train
    print_step(5, total_steps, "Training RL agent (10k timesteps)")
    print("⏱️  This will take approximately 2-5 minutes...")
    train_cmd = f'python train.py --training-config {demo_training_config} --env-config {demo_env_config}'
    if not run_command(train_cmd, 'Training'):
        return False

    # Step 6: Evaluate
    print_step(6, total_steps, "Evaluating model performance")
    eval_cmd = 'python evaluate.py --model ckpts/best_model/best_model.zip --env-config configs/env_demo.yaml --output-dir artifacts/evaluation_demo'
    if not run_command(eval_cmd, 'Evaluation'):
        return False

    # Success!
    print_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("Results saved to:")
    print("  - Model: ckpts/best_model/best_model.zip")
    print("  - Evaluation: artifacts/evaluation_demo/")
    print("  - TensorBoard logs: artifacts/tensorboard/")
    print("\nView results:")
    print("  cat artifacts/evaluation_demo/comparison_table.csv")
    print("  open artifacts/evaluation_demo/rl_agent_trades.png")
    print("\nNext steps:")
    print("  1. Check evaluation results in artifacts/evaluation_demo/")
    print("  2. Try full mode: python quickstart.py --mode full")
    print("  3. Or customize configs and run: python train.py")

    return True

def quickstart_full():
    """Run full pipeline with real data."""
    print_header("QUICK START - FULL MODE")
    print("This will:")
    print("  - Fetch real BTC/USDT data from Binance (365 days)")
    print("  - Train for 200,000 timesteps (~2-4 hours)")
    print("  - Evaluate the model")
    print("  - Generate comprehensive reports")

    print("\n⚠️  Requirements:")
    print("  - Binance API keys in .env file (or use --no-api-key)")
    print("  - ~3-5 hours of time")
    print("  - ~3GB disk space")

    response = input("\nContinue with full training? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return False

    total_steps = 6

    # Step 1: Check dependencies
    print_step(1, total_steps, "Checking dependencies")
    if not check_dependencies():
        return False

    # Step 2: Fetch data
    print_step(2, total_steps, "Fetching BTC/USDT data from Binance")
    print("⏱️  This may take a few minutes...")

    # Check if .env exists
    if not Path('.env').exists():
        print("⚠️  No .env file found. Using public API (limited to recent data)")
        fetch_cmd = 'python fetch_data.py --no-api-key --days 90'
    else:
        print("✅ Using API keys from .env")
        fetch_cmd = 'python fetch_data.py --days 365'

    if not run_command(fetch_cmd, 'Data fetching'):
        return False

    # Step 3: Validate configuration
    print_step(3, total_steps, "Validating configuration")
    if not run_command('python config_validator.py', 'Config validation'):
        return False

    # Step 4: Run tests (optional)
    print_step(4, total_steps, "Running tests (optional)")
    response = input("Run tests? (yes/no, default: no): ")
    if response.lower() in ['yes', 'y']:
        run_command('pytest test_data_manager_pytest.py test_trading_env_pytest.py -v', 'Tests')
    else:
        print("⏭️  Skipping tests")

    # Step 5: Train
    print_step(5, total_steps, "Training RL agent (200k timesteps)")
    print("⏱️  This will take approximately 2-4 hours on CPU...")
    print("💡 Tip: Monitor progress with TensorBoard in another terminal:")
    print("   tensorboard --logdir artifacts/tensorboard/")

    if not run_command('python train.py', 'Training'):
        return False

    # Step 6: Evaluate
    print_step(6, total_steps, "Evaluating model performance")
    if not run_command('python evaluate.py --model ckpts/best_model/best_model.zip', 'Evaluation'):
        return False

    # Success!
    print_header("FULL TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("Results saved to:")
    print("  - Model: ckpts/best_model/best_model.zip")
    print("  - Evaluation: artifacts/evaluation/")
    print("  - TensorBoard logs: artifacts/tensorboard/")
    print("\nView results:")
    print("  cat artifacts/evaluation/comparison_table.csv")
    print("  open artifacts/evaluation/portfolio_comparison.png")
    print("\nNext steps:")
    print("  1. Analyze evaluation results")
    print("  2. Tune hyperparameters in configs/training.yaml")
    print("  3. Try different features in configs/features.yaml")
    print("  4. Experiment with environment settings")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Quick start script for RL Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with sample data (recommended for first try)
  python quickstart.py --mode demo

  # Full training with real Binance data
  python quickstart.py --mode full
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['demo', 'full'],
        required=True,
        help='Quickstart mode: demo (fast) or full (real training)'
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run appropriate mode
    try:
        if args.mode == 'demo':
            success = quickstart_demo()
        else:
            success = quickstart_full()

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
