# ZNE Estimator Test Suite

Comprehensive pytest unit tests for the ZNEEstimator implementation.

## Test Coverage

The test suite contains **49 comprehensive tests** organized into the following categories:

### 1. Initialization Tests (7 tests)
- ✅ Valid parameters initialization
- ✅ Default parameters initialization
- ✅ Diagram saving with custom directory
- ✅ Diagram saving with default path
- ✅ Invalid noise factors validation (too few, < 1, empty)

### 2. Noise Factor Validation (5 tests)
- ✅ Valid odd integer noise factors
- ✅ Invalid even integers rejection
- ✅ Invalid float values rejection
- ✅ Negative values rejection
- ✅ Zero value rejection

### 3. Circuit Folding (4 tests)
- ✅ Folding with noise factor 1 (no folding)
- ✅ Folding with noise factor 3
- ✅ Folding with noise factor 5
- ✅ Qubit preservation during folding

### 4. Extrapolation Methods (5 tests)
- ✅ Linear extrapolation
- ✅ Polynomial extrapolation (with sufficient data points)
- ✅ Exponential extrapolation
- ✅ Exponential fallback to linear for non-positive values
- ✅ Invalid extrapolator error handling

### 5. Result Format & Metadata (5 tests)
- ✅ Basic result structure (PrimitiveResult, PubResult)
- ✅ Data fields (evs, stds, shape)
- ✅ ZNE-specific metadata fields
- ✅ Scaled circuits storage (when enabled)
- ✅ Scaled circuits not stored (when disabled)

### 6. Simulator Integration (3 tests)
- ✅ StatevectorEstimator integration
- ✅ Noisy Aer Estimator integration (with transpiled circuits)
- ✅ Multiple observables support

### 7. Image Generation (4 tests)
- ✅ Diagram generation disabled
- ✅ Diagram generation with default path
- ✅ Diagram generation with custom directory
- ✅ Automatic directory creation

### 8. Default Behavior (1 test)
- ✅ Minimal noise factors (1, 3) functionality

### 9. Runtime Parameter Override (3 tests)
- ✅ Noise factor override at runtime
- ✅ Precision override at runtime
- ✅ Invalid runtime noise factors rejection

### 10. Downstream Compatibility (4 tests)
- ✅ Result iteration
- ✅ Result indexing
- ✅ Numpy array operations on data
- ✅ Job interface compatibility

### 11. Edge Cases (5 tests)
- ✅ Empty pubs list
- ✅ Single-qubit circuits
- ✅ Large circuits (4 qubits)
- ✅ Parameterized circuits
- ✅ Multiple parameter sets

### 12. Numerical Stability (3 tests)
- ✅ Extrapolation with very small standard deviations
- ✅ Extrapolation with large standard deviations
- ✅ Extrapolation with zero standard deviations

## Running the Tests

### Run all tests:
```bash
pytest tests/test_zne_estimator.py -v
```

### Run specific test class:
```bash
pytest tests/test_zne_estimator.py::TestInitialization -v
```

### Run specific test:
```bash
pytest tests/test_zne_estimator.py::TestInitialization::test_init_valid_parameters -v
```

### Run with coverage:
```bash
pytest tests/test_zne_estimator.py --cov=zne_estimator --cov-report=html
```

## Test Fixtures

### `simple_circuit`
A Bell state circuit (H gate + CNOT) used for basic testing.

### `simple_observable`
A ZZ Pauli observable for expectation value measurements.

### `statevector_estimator`
A StatevectorEstimator instance for ideal (noiseless) simulations.

### `noisy_estimator`
An Aer EstimatorV2 instance for testing with realistic quantum noise.

### `temp_diagram_dir`
A temporary directory for testing circuit diagram generation, automatically cleaned up.

## Key Test Scenarios

### 1. Downstream Consumption Tests
The test suite verifies that ZNEEstimator results can be consumed by:
- Standard iteration and indexing operations
- Numpy array operations
- Any component expecting standard Qiskit PrimitiveResult format

### 2. Wrong Values & Error Handling
Tests validate proper error handling for:
- Invalid noise factors (even numbers, floats, negatives, too few)
- Invalid extrapolation methods
- Empty or malformed inputs

### 3. Simulator Compatibility
Tests verify functionality with:
- StatevectorEstimator (ideal, noiseless)
- Aer EstimatorV2 (noisy, realistic)
- Transpiled circuits (ISA-compliant)

### 4. Image Generation
Tests verify:
- Circuit diagrams saved correctly with enabled flag
- No diagrams generated when flag is disabled
- Custom directory support and creation
- Proper error handling when diagram generation fails

### 5. Flag Behavior
Tests for various configuration flags:
- `store_scaled_circuits`: Controls memory usage by optionally storing folded circuits
- `save_circuit_diagrams`: Enables/disables diagram generation
- `diagram_output_dir`: Custom path for saving diagrams

## Bugs Fixed During Testing

### 1. Path Import Bug
**Issue**: Code was importing `Path` from `zipfile` instead of `pathlib`
**Fix**: Changed import to use `pathlib.Path`
**Impact**: Diagram directory creation was failing

### 2. Custom Instruction Issue
**Issue**: `make_folded_block()` was creating custom instructions not recognized by Aer
**Fix**: Removed custom instruction, inline fold circuits using compose
**Impact**: ZNE now works with all Qiskit simulators

### 3. Covariance Calculation with Minimal Data Points
**Issue**: `np.polyfit` with `cov=True` requires `degree + 2` data points
**Fix**: Added fallback to residual-based std estimation when insufficient points
**Impact**: ZNE now works with minimal (2) noise factors

## Test Requirements

The tests require the following packages (specified in `requirements.txt`):
- qiskit >= 1.0.0
- qiskit-aer >= 0.17.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0

## Test Configuration

Tests are configured via `pytest.ini` with:
- Verbose output by default
- Short traceback format
- Warnings suppressed
- Non-interactive matplotlib backend (Agg)

## Coverage Summary

✅ **49/49 tests passing (100%)**

The test suite provides comprehensive coverage of:
- All public methods and parameters
- Error handling and edge cases
- Integration with Qiskit simulators
- Result format compatibility
- Numerical stability
- Configuration options and flags
