#!/usr/bin/env python3
"""
Explore gapless-crypto-data package capabilities
"""
import sys
sys.path.append('user_strategies')

# Explore the gapless-crypto-data package
try:
    import gapless_crypto_data as gcd

    print("📦 Gapless Crypto Data Package Info:")
    print(f"Version: {gcd.__version__ if hasattr(gcd, '__version__') else 'Unknown'}")
    print(f"Available attributes: {[attr for attr in dir(gcd) if not attr.startswith('_')]}")

    # Try to understand the API
    if hasattr(gcd, 'fetch'):
        print(f"\nfetch function signature: {gcd.fetch.__doc__}")

    if hasattr(gcd, 'get_data'):
        print(f"\nget_data function signature: {gcd.get_data.__doc__}")

    # Try to get some sample data
    print("\n🔍 Attempting to fetch sample crypto data...")

    # Common crypto pairs to test
    test_pairs = ['BTCUSDT', 'ETHUSDT', 'BTCUSD']

    for pair in test_pairs:
        try:
            print(f"\nTesting {pair}:")

            # Try different possible function names
            for func_name in ['fetch', 'get_data', 'load_data', 'download']:
                if hasattr(gcd, func_name):
                    func = getattr(gcd, func_name)
                    print(f"  Found function: {func_name}")

                    # Try to call with minimal parameters
                    try:
                        # Test with recent timeframe
                        result = func(pair, start='2024-01-01', end='2024-01-02')
                        print(f"  ✅ {func_name}({pair}) successful!")
                        print(f"  Result type: {type(result)}")
                        if hasattr(result, 'shape'):
                            print(f"  Shape: {result.shape}")
                        if hasattr(result, 'columns'):
                            print(f"  Columns: {list(result.columns)}")
                        print(f"  First few rows:\n{result.head() if hasattr(result, 'head') else result}")
                        break  # Success, no need to try other functions
                    except Exception as e:
                        print(f"  ❌ {func_name}({pair}) failed: {e}")
            break  # If we got data for one pair, that's enough for exploration
        except Exception as e:
            print(f"❌ Failed to test {pair}: {e}")

except ImportError as e:
    print(f"❌ Failed to import gapless-crypto-data: {e}")
except Exception as e:
    print(f"❌ Error exploring gapless-crypto-data: {e}")