"""ClickHouse connection configuration for triple-barrier backtesting.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

This module provides ClickHouse connection configuration with environment
variable support and local fallbacks. Uses gapless-crypto-clickhouse as
the primary data source with zero-gap guarantees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClickHouseConfig:
    """Immutable ClickHouse connection configuration.

    Attributes:
        host: ClickHouse server hostname
        port: ClickHouse server port (default: 8443 for Cloud, 9000 for local)
        user: Database username
        password: Database password
        database: Database name
        secure: Use HTTPS/TLS connection
    """

    host: str
    port: int
    user: str
    password: str
    database: str
    secure: bool = True

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        """Load configuration from environment variables.

        Environment variables:
            CLICKHOUSE_HOST: Server hostname (default: localhost)
            CLICKHOUSE_PORT: Server port (default: 8443)
            CLICKHOUSE_USER: Username (default: default)
            CLICKHOUSE_PASSWORD: Password (default: empty)
            CLICKHOUSE_DB: Database name (default: default)
            CLICKHOUSE_SECURE: Use TLS (default: true)

        Returns:
            ClickHouseConfig instance with environment values
        """
        return cls(
            host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
            port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
            user=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            database=os.environ.get("CLICKHOUSE_DB", "default"),
            secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
        )

    def to_dict(self) -> dict[str, str | int | bool]:
        """Convert to dictionary for clickhouse-driver connection.

        Returns:
            Dictionary with connection parameters
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "secure": self.secure,
        }


# Default configuration loaded from environment
DEFAULT_CONFIG = ClickHouseConfig.from_env()
