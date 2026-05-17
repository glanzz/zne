# Zero Noise Extrapolation (ZNE) Estimator for Qiskit

A custom Qiskit Estimator implementing Zero Noise Extrapolation (ZNE), a quantum error mitigation technique that improves expectation value estimates by amplifying and extrapolating circuit noise.

## Overview

The estimator class provides a drop-in replacement for standard Qiskit estimators, with additional parameters to enable ZNE. It supports both global and local folding methods for noise scaling, and multiple extrapolation techniques (polynomial, linear, exponential). The implementation is designed to be flexible and compatible with various backends, including ideal simulators and noisy density matrix simulators. The estimator also includes options for saving circuit diagrams and detailed metadata in the results for analysis. The estimator also avoids further transpilation after folding to preserve the noise scaling, and includes comprehensive error handling for invalid parameters.

## ZNE Variant

This implementation uses **digital ZNE with unitary folding**:

### Noise Scaling Methods

1. **Global Folding** (default)
   - Transforms entire circuit: `U → U(U†U)^n`
   - Scale factor: `λ = 1 + 2n` (must be odd integers: 1, 3, 5, 7, ...)
   - Simpler and more predictable
   - Example: For λ=3, circuit becomes `U·U†·U`

2. **Local Folding**
   - Folds individual gates: `G → G(G†G)^n`
   - Supports non-integer scale factors
   - Prioritizes 2-qubit gates (typically noisier)
   - More flexible but more complex

### Extrapolation Methods

1. **Polynomial** - Custom degree polynomial fitting
2. **Linear** - Simple linear extrapolation
3. **Exponential** - Fits exponential decay model

## Installation
```bash
python3 -m venv zne-env
source zne-env/bin/activate  # On Windows: zne-env\Scripts\activate
pip install -r requirements.txt
```


## Tests
```bash
pytest tests/test_zne_estimator.py -v
```


## Parameters

### `ZNEEstimator.__init__()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_estimator` | `BaseEstimatorV2` | *required* | The underlying Qiskit estimator to use for circuit execution |
| `extrapolator` | `str` | `'linear'` | Extrapolation method: `'linear'`, `'polynomial'`, or `'exponential'` |
| `folding` | `str` | `'global'` | Folding strategy: `'global'` or `'local'` |
| `default_precision` | `float` | `0.0` | Default target precision for expectation value estimates |
| `store_scaled_circuits` | `bool` | `True` | Whether to store scaled circuits in result metadata |
| `save_circuit_diagrams` | `bool` | `True` | Whether to save circuit diagrams as PNG files |
| `diagram_output_dir` | `Optional[Union[str, Path]]` | `'./zne_diagrams'` | Directory for saving circuit diagrams |

### `ZNEEstimator.run()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pubs` | `Iterable[EstimatorPubLike]` | *required* | Iterable of pub-like objects (circuit, observable, parameters) |
| `precision` | `Optional[float]` | `None` | Target precision for pubs that don't set their own. If `None`, uses `default_precision` from `__init__` |
| `noise_factors` | `Optional[Sequence[float]]` | `None` | Noise scale factors for ZNE, e.g., `[1, 3, 5]`. If `None`, ZNE is disabled and the base estimator is used directly. For global folding, must be odd integers (1, 3, 5, ...). |

## Result Object

The estimator returns a `PrimitiveResult[PubResult]` object with the following structure:

```python
result = zne_est.run(pubs, noise_factors=[1, 3, 5]).result()

# Access individual pub results
pub_result = result[0]  # PubResult for first pub

# Access ZNE-mitigated expectation values and standard deviations
print(pub_result.data.evs)   # numpy array of extrapolated expectation values
print(pub_result.data.stds)  # numpy array of extrapolated standard deviations

# Access ZNE metadata for debugging and analysis
zne_metadata = pub_result.metadata['zne']  # dict with the following keys:
# - 'noise_factors': list[float] - scale factors used (e.g., [1, 3, 5])
# - 'extrapolator': str - extrapolation method used ('linear', 'polynomial', 'exponential')
# - 'raw_pub_results': list[PubResult] - full PubResult objects from base estimator for each noise factor
# - 'noisy_evs': ndarray - raw expectation values at each noise level, shape (num_factors, ...)
# - 'noisy_stds': ndarray - raw standard deviations at each noise level, shape (num_factors, ...)
# - 'target_precision': float - target precision used for the estimation
# - 'scaled_circuits': list[QuantumCircuit] - folded circuits (if store_scaled_circuits=True)
# - 'circuit_diagram_paths': list[str] - paths to saved circuit diagrams (if save_circuit_diagrams=True)

# Access top-level metadata
result.metadata['zne']  # True - indicates ZNE was applied
result.metadata['noise_factors']  # list of noise factors used
result.metadata['job_id']  # unique job identifier
```

## How It Works

### 1. Noise Scaling (Circuit Folding)

ZNE amplifies noise by adding identity operations to the circuit:

**Global Folding:**
```
Original:  U = H-CNOT-H
λ=1:       H-CNOT-H                    (original)
λ=3:       H-CNOT-H-H-CNOT†-H-H-CNOT-H (U·U†·U)
λ=5:       [U·U†·U]·U†·U               (more folding)
```

**Local Folding:**
```
Original:  H-CNOT-H
Fold CNOT: H-CNOT-CNOT†-CNOT-H
```

### 2. Measurement

Execute each folded circuit and measure the observable:
```
λ=1: E(1) = measured expectation value
λ=3: E(3) = measured expectation value (more noisy)
λ=5: E(5) = measured expectation value (even more noisy)
```

### 3. Extrapolation

Fit a curve through `(λ, E(λ))` points and extrapolate to λ=0:

```
Polynomial: E(λ) = a₀ + a₁λ + a₂λ² + ..., return a₀
Linear:     E(λ) = aλ + b, return b
Exponential: E(λ) = a + b·exp(-cλ), return a + b
```

## Implementation Details

### Key Design Decisions

1. **Extends `BaseEstimatorV1`**: Ensures full interface compatibility
2. **Transpile before folding**: Optimization happens once, then folding preserves gates
3. **Disable optimization after folding**: Prevents compiler from undoing the folding
4. **Metadata in results**: Includes intermediate values for debugging and analysis
5. **Multiple extrapolation methods**: Different noise models benefit from different fits


## Limitations

1. **Digital ZNE only**: Requires gate-level control, not pulse-level
2. **Assumes gate-independent noise**: Works best with uniform noise models
3. **Circuit depth scaling**: Very deep circuits may become impractical at high scale factors
4. **Extrapolation accuracy**: Depends on noise model and scale factors chosen

## References

1. Temme et al., "Error Mitigation for Short-Depth Quantum Circuits", *Nature* 567, 491–495 (2019)
2. Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum error mitigation", [arXiv:2005.10921](https://arxiv.org/pdf/2005.10921)
3. LaRose et al., "Mitiq: A software package for error mitigation on noisy quantum computers"

## License

This implementation is provided for educational and research purposes.

## Contributing

For bugs or improvements, please submit an issue or pull request.

