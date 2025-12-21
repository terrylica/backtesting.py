# Strategy Configurations

This directory contains configuration files for trading strategies.

## Purpose

- Store strategy parameters and settings
- Environment-specific configurations (development, testing, production)
- Data source configurations
- Risk management parameters

## Usage

Configuration files should be in JSON or YAML format for easy parsing:

```python
import json
from pathlib import Path

config_path = Path(__file__).parent / "configs" / "ml_strategy_config.json"
with open(config_path) as f:
    config = json.load(f)
```

## Examples

- `ml_strategy_config.json` - ML walk-forward strategy parameters
- `data_sources.yaml` - Data source configurations (EURUSD, crypto pairs)
- `risk_management.json` - Position sizing and risk parameters
