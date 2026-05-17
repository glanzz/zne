# ZNE Estimator Test Suite

Comprehensive pytest tests validating ZNEEstimator's core functionality across 12 categories: initialization with various parameters, noise factor validation (odd integers for global folding), circuit folding methods (global and local), extrapolation techniques (linear, polynomial, exponential), result format compatibility, simulator integration (StatevectorEstimator and Aer), circuit diagram generation, runtime parameter overrides, downstream compatibility, edge cases (parameterized circuits, empty pubs), and numerical stability. Tests ensure proper error handling and verify compatibility with Qiskit's PrimitiveResult interface.

**Run tests:** `pytest tests/test_zne_estimator.py -v`
