"""
Pytest configuration and shared fixtures for ZNE Estimator tests.
"""

import pytest
import warnings
import matplotlib


def pytest_configure(config):
    """Configure pytest."""
    # Use non-interactive matplotlib backend for tests
    matplotlib.use('Agg')

    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "noisy: marks tests that use noisy simulators"
    )


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Qiskit.*")
